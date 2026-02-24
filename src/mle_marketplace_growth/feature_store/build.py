"""Build feature-store datasets, run DQ checks, and write a run manifest for lineage and downstream experiment tracking."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import duckdb
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "duckdb is required. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


REQUIRED_SOURCE_COLUMNS = {
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
}

def _load_sql(sql_path: Path) -> str:
    return sql_path.read_text(encoding="utf-8")


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _copy_table_to_csv(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    output_path: Path,
    copy_sql_template: str,
    count_rows_sql_template: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quoted_path = _sql_quote(str(output_path))
    copy_sql = copy_sql_template.replace("{table_name}", table_name).replace("{output_path}", quoted_path)
    connection.execute(copy_sql)
    count_rows_sql = count_rows_sql_template.replace("{table_name}", table_name)
    row_count = connection.execute(count_rows_sql).fetchone()[0]
    return int(row_count)


def _run_dq_check(connection: duckdb.DuckDBPyConnection, sql: str, error_message: str) -> None:
    failing_rows = int(connection.execute(sql).fetchone()[0] or 0)
    if failing_rows > 0:
        raise ValueError(error_message)


def main() -> None:
    """Run the feature-store build pipeline."""
    # CLI arguments and input paths.
    parser = argparse.ArgumentParser(description="Build local feature-store tables with DuckDB SQL.")
    parser.add_argument(
        "--input-csv",
        default="data/bronze/online_retail_ii/raw.csv",
        help="Path to raw Online Retail II CSV",
    )
    parser.add_argument(
        "--output-root",
        default="data",
        help="Output root path for silver/gold materializations",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="As-of date for feature snapshots in YYYY-MM-DD (default: max event date in source)",
    )
    parser.add_argument(
        "--bad-ts-threshold",
        type=float,
        default=0.01,
        help="Maximum tolerated ratio of rows with unparsable InvoiceDate before failing",
    )
    parser.add_argument(
        "--split-version",
        default="time_rank_v1",
        help="Identifier for the split strategy/version written into split artifacts",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_root = Path(args.output_root)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Load SQL assets (pipeline steps, model transformations, DQ checks).
    sql_dir = Path(__file__).resolve().parent / "sql"

    # Pipeline SQL.
    create_raw_source_sql_template = _load_sql(sql_dir / "pipeline" / "create_raw_source.sql")
    raw_bad_timestamp_counts_sql = _load_sql(sql_dir / "pipeline" / "raw_bad_timestamp_counts.sql")
    raw_exact_duplicate_rows_sql = _load_sql(sql_dir / "pipeline" / "raw_exact_duplicate_rows_count.sql")
    copy_table_to_csv_sql_template = _load_sql(sql_dir / "pipeline" / "copy_table_to_csv.sql")
    count_rows_sql_template = _load_sql(sql_dir / "pipeline" / "count_rows.sql")
    describe_raw_source_sql = _load_sql(sql_dir / "pipeline" / "describe_raw_source.sql")
    max_silver_event_date_sql = _load_sql(sql_dir / "pipeline" / "max_silver_event_date.sql")

    # Model SQL (silver + gold).
    silver_sql = _load_sql(sql_dir / "silver" / "transactions_line_items.sql")
    interactions_sql = _load_sql(sql_dir / "gold" / "recommender" / "interaction_events.sql")
    user_item_splits_sql_template = _load_sql(sql_dir / "gold" / "recommender" / "user_item_splits.sql")
    labels_sql_template = _load_sql(sql_dir / "gold" / "growth_uplift" / "labels.sql")
    user_features_sql_template = _load_sql(sql_dir / "gold" / "growth_uplift" / "user_features_asof.sql")
    uplift_train_sql_template = _load_sql(sql_dir / "gold" / "growth_uplift" / "uplift_train_dataset.sql")

    # DQ SQL.
    dq_invalid_split_sql = _load_sql(sql_dir / "dq" / "gold_invalid_split_count.sql")
    dq_split_chronology_sql = _load_sql(sql_dir / "dq" / "gold_split_chronology_violation_count.sql")
    dq_user_features_duplicates_sql = _load_sql(sql_dir / "dq" / "gold_user_features_duplicate_grain_count.sql")
    dq_invalid_labels_sql = _load_sql(sql_dir / "dq" / "gold_invalid_label_rows_count.sql")
    dq_uplift_train_duplicates_sql = _load_sql(sql_dir / "dq" / "gold_uplift_train_duplicate_grain_count.sql")

    silver_path = output_root / "silver" / "transactions_line_items" / "transactions_line_items.csv"
    gold_root = output_root / "gold" / "feature_store"
    recommender_root = gold_root / "recommender"
    growth_uplift_root = gold_root / "growth_uplift"

    # Bootstrap raw source and validate source-level quality.
    connection = duckdb.connect(database=":memory:")
    quoted_input_csv = _sql_quote(str(input_csv))
    create_raw_source_sql = create_raw_source_sql_template.replace("{input_csv}", quoted_input_csv)
    connection.execute(create_raw_source_sql)

    source_schema = connection.execute(describe_raw_source_sql).fetchall()
    available_columns = {row[0] for row in source_schema}
    missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - available_columns)
    if missing_columns:
        raise ValueError(f"Missing required source columns: {missing_columns}")

    timestamp_quality = connection.execute(raw_bad_timestamp_counts_sql).fetchone()
    total_rows = int(timestamp_quality[0] or 0)
    bad_rows = int(timestamp_quality[1] or 0)
    exact_duplicate_rows = int(connection.execute(raw_exact_duplicate_rows_sql).fetchone()[0] or 0)
    bad_ratio = (bad_rows / total_rows) if total_rows else 0.0
    if bad_ratio > args.bad_ts_threshold:
        raise ValueError(
            f"Timestamp parse failure ratio {bad_ratio:.4f} exceeded threshold {args.bad_ts_threshold:.4f}"
        )

    # Build silver then resolve feature snapshot date.
    connection.execute(silver_sql)

    if args.as_of_date:
        as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
    else:
        as_of_value = connection.execute(max_silver_event_date_sql).fetchone()[0]
        if as_of_value is None:
            raise ValueError("No rows in silver_transactions_line_items; cannot resolve as_of_date")
        as_of_date = as_of_value
    as_of_partition = f"as_of_date={as_of_date.isoformat()}"
    labels_path = growth_uplift_root / "labels" / as_of_partition / "labels.csv"
    features_path = growth_uplift_root / "user_features_asof" / as_of_partition / "user_features_asof.csv"
    uplift_train_path = growth_uplift_root / "uplift_train_dataset" / as_of_partition / "uplift_train_dataset.csv"
    manifest_path = growth_uplift_root / "_meta" / as_of_partition / "run_manifest.json"
    interactions_path = recommender_root / "interaction_events" / "interaction_events.csv"
    user_item_splits_path = recommender_root / "user_item_splits" / "user_item_splits.csv"

    # Build gold models: growth uplift engine.
    user_features_sql = user_features_sql_template.replace("{as_of_date}", as_of_date.isoformat())
    labels_sql = labels_sql_template.replace("{as_of_date}", as_of_date.isoformat())
    uplift_train_sql = uplift_train_sql_template.replace("{as_of_date}", as_of_date.isoformat())

    # Build gold models: recommender engine.
    user_item_splits_sql = user_item_splits_sql_template.replace("{split_version}", args.split_version)
    for sql in [interactions_sql, user_item_splits_sql, labels_sql, user_features_sql, uplift_train_sql]:
        connection.execute(sql)

    # Run dataset-level DQ checks.
    dq_checks = [
        # Recommender gold checks.
        (dq_invalid_split_sql, "DQ_GOLD_005 failed: invalid split value in gold_user_item_splits"),
        (dq_split_chronology_sql, "DQ_GOLD_006 failed: chronology violation in gold_user_item_splits"),
        # Growth uplift gold checks.
        (dq_user_features_duplicates_sql, "DQ_GOLD_001 failed: duplicate (user_id, as_of_date) in gold_user_features_asof"),
        (dq_invalid_labels_sql, "DQ_GOLD_003 failed: invalid label metadata in gold_labels"),
        (dq_uplift_train_duplicates_sql, "DQ_GOLD_009 failed: duplicate (user_id, as_of_date) in gold_uplift_train_dataset"),
    ]
    for dq_sql, dq_error in dq_checks:
        _run_dq_check(connection, dq_sql, dq_error)

    # Materialize outputs and collect row counts.
    artifacts = [
        ("silver_transactions_line_items", "silver", "silver_transactions_line_items", silver_path),
        # Recommender engine outputs.
        ("gold_interaction_events", "gold", "gold_interaction_events", interactions_path),
        ("gold_user_item_splits", "gold", "gold_user_item_splits", user_item_splits_path),
        # Growth uplift engine outputs.
        ("gold_labels", "gold", "gold_labels", labels_path),
        ("gold_user_features_asof", "gold", "gold_user_features_asof", features_path),
        ("gold_uplift_train_dataset", "gold", "gold_uplift_train_dataset", uplift_train_path),
    ]
    artifact_rows = {}
    for artifact_name, layer, table_name, artifact_path in artifacts:
        row_count = _copy_table_to_csv(
            connection,
            table_name,
            artifact_path,
            copy_table_to_csv_sql_template,
            count_rows_sql_template,
        )
        artifact_rows[artifact_name] = row_count
        print(f"Wrote {layer} table: {artifact_path} ({row_count} rows)")

    # Write run manifest.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input": {"path": str(input_csv)},
        "params": {
            "as_of_date": as_of_date.isoformat(),
            "bad_ts_threshold": args.bad_ts_threshold,
            "split_version": args.split_version,
        },
        "quality": {
            "raw_total_rows": total_rows,
            "raw_exact_duplicate_rows": exact_duplicate_rows,
            "raw_bad_timestamp_rows": bad_rows,
            "raw_bad_timestamp_ratio": round(bad_ratio, 6),
        },
        "artifacts": {
            "silver_transactions_line_items": {
                "path": str(silver_path),
                "rows": artifact_rows["silver_transactions_line_items"],
            },
            # Recommender engine artifacts.
            "gold_interaction_events": {
                "path": str(interactions_path),
                "rows": artifact_rows["gold_interaction_events"],
            },
            "gold_user_item_splits": {
                "path": str(user_item_splits_path),
                "rows": artifact_rows["gold_user_item_splits"],
            },
            # Growth uplift engine artifacts.
            "gold_labels": {"path": str(labels_path), "rows": artifact_rows["gold_labels"]},
            "gold_user_features_asof": {
                "path": str(features_path),
                "rows": artifact_rows["gold_user_features_asof"],
            },
            "gold_uplift_train_dataset": {
                "path": str(uplift_train_path),
                "rows": artifact_rows["gold_uplift_train_dataset"],
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
