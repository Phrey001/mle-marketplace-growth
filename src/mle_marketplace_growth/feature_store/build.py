"""Build feature-store datasets, run DQ checks, and write a run manifest for lineage and downstream experiment tracking."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import yaml


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


# ===== SQL + DuckDB Helpers =====
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
    if failing_rows > 0: raise ValueError(error_message)

def _load_yaml_defaults(path_value: str | None, label: str) -> dict:
    if not path_value:
        return {}
    config_path = Path(path_value)
    if not config_path.exists():
        raise FileNotFoundError(f"{label} file not found: {config_path}")
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"{label} file must use .yaml or .yml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} file must contain a key-value object")
    return payload


# ===== Entry Point =====
def main() -> None:
    """Run the feature-store build pipeline."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--shared-config", default=None, help="Optional shared YAML config file")
    pre_args, remaining_argv = pre_parser.parse_known_args()
    shared_defaults = _load_yaml_defaults(pre_args.shared_config, "Shared config")
    cfg = shared_defaults.get

    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Build local feature-store tables with DuckDB SQL.")
    parser.add_argument("--shared-config", default=pre_args.shared_config, help="Optional shared YAML config file")
    parser.add_argument("--input-csv", default=cfg("input_csv", "data/bronze/online_retail_ii/raw.csv"), help="Path to raw Online Retail II CSV")
    parser.add_argument("--output-root", default=cfg("output_root", "data"), help="Output root path for silver/gold materializations")
    parser.add_argument("--as-of-date", default=None, help="As-of date for feature snapshots in YYYY-MM-DD (default: max event date in source)")
    parser.add_argument("--build-engines", default="purchase_propensity,recommender", help="Comma-separated engines to build: shared,purchase_propensity,recommender")
    parser.add_argument("--recommender-min-event-date", default=None, help="Optional lower bound event_date filter for recommender datasets in YYYY-MM-DD")
    parser.add_argument("--recommender-max-event-date", default=None, help="Optional upper bound event_date filter for recommender datasets in YYYY-MM-DD")
    parser.add_argument("--bad-ts-threshold", type=float, default=0.01, help="Maximum tolerated ratio of rows with unparsable InvoiceDate before failing")
    parser.add_argument("--split-version", default="time_rank_v1", help="Identifier for the split strategy/version written into split artifacts")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    build_engines = {value.strip() for value in args.build_engines.split(",") if value.strip()}
    supported_engines = {"purchase_propensity", "recommender", "shared"}
    unknown_engines = sorted(build_engines - supported_engines)
    if unknown_engines: raise ValueError(f"Unsupported engine(s) in --build-engines: {unknown_engines}")
    if not build_engines: raise ValueError("--build-engines must include at least one supported engine")
    if "shared" in build_engines and len(build_engines) > 1:
        raise ValueError("--build-engines=shared must be run as a standalone shared-layer build command")
    build_shared = build_engines == {"shared"}
    input_csv = Path(args.input_csv)
    if build_shared and not input_csv.exists(): raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # ===== Load SQL Assets =====
    sql_dir = Path(__file__).resolve().parent / "sql"

    # Pipeline SQL
    create_raw_source_sql_template = _load_sql(sql_dir / "pipeline" / "create_raw_source.sql")
    raw_bad_timestamp_counts_sql = _load_sql(sql_dir / "pipeline" / "raw_bad_timestamp_counts.sql")
    raw_exact_duplicate_rows_sql = _load_sql(sql_dir / "pipeline" / "raw_exact_duplicate_rows_count.sql")
    copy_table_to_csv_sql_template = _load_sql(sql_dir / "pipeline" / "copy_table_to_csv.sql")
    count_rows_sql_template = _load_sql(sql_dir / "pipeline" / "count_rows.sql")
    describe_raw_source_sql = _load_sql(sql_dir / "pipeline" / "describe_raw_source.sql")
    max_silver_event_date_sql = _load_sql(sql_dir / "pipeline" / "max_silver_event_date.sql")

    # Model SQL (silver + gold)
    silver_sql = _load_sql(sql_dir / "silver" / "transactions_line_items.sql")
    interactions_sql_template = _load_sql(sql_dir / "gold" / "recommender" / "interaction_events.sql")
    user_item_splits_sql_template = _load_sql(sql_dir / "gold" / "recommender" / "user_item_splits.sql")
    user_index_sql = _load_sql(sql_dir / "gold" / "recommender" / "user_index.sql")
    item_index_sql = _load_sql(sql_dir / "gold" / "recommender" / "item_index.sql")
    labels_sql_template = _load_sql(sql_dir / "gold" / "purchase_propensity" / "labels.sql")
    user_features_sql_template = _load_sql(sql_dir / "gold" / "purchase_propensity" / "user_features_asof.sql")
    propensity_train_sql_template = _load_sql(sql_dir / "gold" / "purchase_propensity" / "propensity_train_dataset.sql")

    # DQ SQL
    dq_invalid_split_sql = _load_sql(sql_dir / "dq" / "gold_invalid_split_count.sql")
    dq_split_chronology_sql = _load_sql(sql_dir / "dq" / "gold_split_chronology_violation_count.sql")
    dq_user_features_duplicates_sql = _load_sql(sql_dir / "dq" / "gold_user_features_duplicate_grain_count.sql")
    dq_invalid_labels_sql = _load_sql(sql_dir / "dq" / "gold_invalid_label_rows_count.sql")
    dq_propensity_train_duplicates_sql = _load_sql(sql_dir / "dq" / "gold_propensity_train_duplicate_grain_count.sql")

    silver_path = output_root / "silver" / "transactions_line_items" / "transactions_line_items.csv"
    gold_root = output_root / "gold" / "feature_store"
    recommender_root = gold_root / "recommender"
    propensity_root = gold_root / "purchase_propensity"

    # ===== Bootstrap Source + Source DQ =====
    connection = duckdb.connect(database=":memory:")
    total_rows = bad_rows = exact_duplicate_rows = 0
    bad_ratio = 0.0
    if build_shared:
        quoted_input_csv = _sql_quote(str(input_csv))
        create_raw_source_sql = create_raw_source_sql_template.replace("{input_csv}", quoted_input_csv)
        connection.execute(create_raw_source_sql)

        source_schema = connection.execute(describe_raw_source_sql).fetchall()
        available_columns = {row[0] for row in source_schema}
        missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - available_columns)
        if missing_columns: raise ValueError(f"Missing required source columns: {missing_columns}")

        timestamp_quality = connection.execute(raw_bad_timestamp_counts_sql).fetchone()
        total_rows = int(timestamp_quality[0] or 0)
        bad_rows = int(timestamp_quality[1] or 0)
        exact_duplicate_rows = int(connection.execute(raw_exact_duplicate_rows_sql).fetchone()[0] or 0)
        bad_ratio = (bad_rows / total_rows) if total_rows else 0.0
        if bad_ratio > args.bad_ts_threshold: raise ValueError(f"Timestamp parse failure ratio {bad_ratio:.4f} exceeded threshold {args.bad_ts_threshold:.4f}")
        connection.execute(silver_sql)
    else:
        if not silver_path.exists():
            raise FileNotFoundError(
                f"Shared silver CSV not found: {silver_path}. "
                "Run feature_store.build with --build-engines shared first."
            )
        connection.execute(
            "CREATE OR REPLACE TEMP TABLE silver_transactions_line_items AS "
            "SELECT "
            "  trim(CAST(invoice_id AS VARCHAR)) AS invoice_id, "
            "  trim(CAST(item_id AS VARCHAR)) AS item_id, "
            "  nullif(trim(CAST(item_description AS VARCHAR)), '') AS item_description, "
            "  CAST(try_cast(trim(CAST(quantity AS VARCHAR)) AS DOUBLE) AS INTEGER) AS quantity, "
            "  try_strptime(trim(CAST(event_ts AS VARCHAR)), '%Y-%m-%d %H:%M:%S') AS event_ts, "
            "  try_cast(trim(CAST(event_date AS VARCHAR)) AS DATE) AS event_date, "
            "  try_cast(trim(CAST(unit_price AS VARCHAR)) AS DOUBLE) AS unit_price, "
            "  try_cast(trim(CAST(line_revenue AS VARCHAR)) AS DOUBLE) AS line_revenue, "
            "  trim(CAST(user_id AS VARCHAR)) AS user_id, "
            "  trim(CAST(country AS VARCHAR)) AS country "
            "FROM read_csv_auto(?, header=TRUE, all_varchar=TRUE, ignore_errors=TRUE)",
            [str(silver_path)],
        )

    propensity_as_of_date = None
    labels_path = None
    features_path = None
    propensity_train_path = None
    build_purchase = "purchase_propensity" in build_engines
    build_recommender = "recommender" in build_engines
    silver_date_bounds = connection.execute("SELECT min(event_date), max(event_date) FROM silver_transactions_line_items").fetchone()
    silver_min_date, silver_max_date = silver_date_bounds[0], silver_date_bounds[1]
    if silver_min_date is None or silver_max_date is None:
        raise ValueError("No rows in silver_transactions_line_items; cannot build gold layers")

    if build_purchase:
        if args.as_of_date:
            propensity_as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
        else:
            as_of_value = connection.execute(max_silver_event_date_sql).fetchone()[0]
            if as_of_value is None: raise ValueError("No rows in silver_transactions_line_items; cannot resolve as_of_date")
            propensity_as_of_date = as_of_value
        if propensity_as_of_date < silver_min_date or propensity_as_of_date > silver_max_date:
            raise ValueError(
                f"Purchase propensity as_of_date {propensity_as_of_date} is outside available silver event_date bounds "
                f"[{silver_min_date}, {silver_max_date}]"
            )
        as_of_partition = f"as_of_date={propensity_as_of_date.isoformat()}"
        labels_path = propensity_root / "labels" / as_of_partition / "labels.csv"
        features_path = propensity_root / "user_features_asof" / as_of_partition / "user_features_asof.csv"
        propensity_train_path = propensity_root / "propensity_train_dataset" / as_of_partition / "propensity_train_dataset.csv"
        manifest_path = propensity_root / "_meta" / as_of_partition / "run_manifest.json"
    else:
        manifest_path = gold_root / "_meta" / "run_manifest.json"
    interactions_path = recommender_root / "interaction_events" / "interaction_events.csv"
    user_item_splits_path = recommender_root / "user_item_splits" / "user_item_splits.csv"
    user_index_path = recommender_root / "user_index" / "user_index.csv"
    item_index_path = recommender_root / "item_index" / "item_index.csv"

    # ===== Build Gold: Purchase Propensity =====
    if "purchase_propensity" in build_engines:
        user_features_sql = user_features_sql_template.replace("{as_of_date}", propensity_as_of_date.isoformat())
        labels_sql = labels_sql_template.replace("{as_of_date}", propensity_as_of_date.isoformat())
        propensity_train_sql = propensity_train_sql_template.replace("{as_of_date}", propensity_as_of_date.isoformat())
        for sql in [labels_sql, user_features_sql, propensity_train_sql]:
            connection.execute(sql)

    # ===== Build Gold: Recommender =====
    if build_recommender:
        recommender_time_filters = []
        recommender_min_date = silver_min_date
        recommender_max_date = silver_max_date
        if args.recommender_min_event_date:
            recommender_min_date = datetime.strptime(args.recommender_min_event_date, "%Y-%m-%d").date()
            recommender_time_filters.append(f"AND event_date >= CAST('{args.recommender_min_event_date}' AS DATE)")
        if args.recommender_max_event_date:
            recommender_max_date = datetime.strptime(args.recommender_max_event_date, "%Y-%m-%d").date()
            recommender_time_filters.append(f"AND event_date <= CAST('{args.recommender_max_event_date}' AS DATE)")
        if recommender_min_date < silver_min_date or recommender_max_date > silver_max_date:
            raise ValueError(
                f"Recommender event-date bounds [{recommender_min_date}, {recommender_max_date}] exceed available silver bounds "
                f"[{silver_min_date}, {silver_max_date}]"
            )
        interactions_sql = interactions_sql_template.replace(
            "{recommender_time_filters}",
            "\n  ".join(recommender_time_filters) if recommender_time_filters else "",
        )
        user_item_splits_sql = user_item_splits_sql_template.replace("{split_version}", args.split_version)
        for sql in [interactions_sql, user_item_splits_sql, user_index_sql, item_index_sql]:
            connection.execute(sql)

    # ===== Run Dataset-Level DQ Checks =====
    dq_checks = []
    if build_recommender:
        dq_checks.extend(
            [
                (dq_invalid_split_sql, "DQ_GOLD_005 failed: invalid split value in gold_user_item_splits"),
                (dq_split_chronology_sql, "DQ_GOLD_006 failed: chronology violation in gold_user_item_splits"),
            ]
        )
    if build_purchase:
        dq_checks.extend(
            [
                (dq_user_features_duplicates_sql, "DQ_GOLD_001 failed: duplicate (user_id, as_of_date) in gold_user_features_asof"),
                (dq_invalid_labels_sql, "DQ_GOLD_003 failed: invalid label metadata in gold_labels"),
                (dq_propensity_train_duplicates_sql, "DQ_GOLD_009 failed: duplicate (user_id, as_of_date) in gold_propensity_train_dataset"),
            ]
        )
    for dq_sql, dq_error in dq_checks:
        _run_dq_check(connection, dq_sql, dq_error)

    # ===== Materialize Outputs =====
    artifacts = []
    if build_shared:
        artifacts.append(("silver_transactions_line_items", "silver", "silver_transactions_line_items", silver_path))
    if build_recommender:
        artifacts.extend(
            [
                ("gold_interaction_events", "gold", "gold_interaction_events", interactions_path),
                ("gold_user_item_splits", "gold", "gold_user_item_splits", user_item_splits_path),
                ("gold_recommender_user_index", "gold", "gold_recommender_user_index", user_index_path),
                ("gold_recommender_item_index", "gold", "gold_recommender_item_index", item_index_path),
            ]
        )
    if build_purchase:
        artifacts.extend(
            [
                ("gold_labels", "gold", "gold_labels", labels_path),
                ("gold_user_features_asof", "gold", "gold_user_features_asof", features_path),
                ("gold_propensity_train_dataset", "gold", "gold_propensity_train_dataset", propensity_train_path),
            ]
        )
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

    # ===== Write Run Manifest =====
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input": {"path": str(input_csv)},
        "params": {
            "as_of_date": propensity_as_of_date.isoformat() if propensity_as_of_date else None,
            "build_engines": sorted(build_engines),
            "recommender_min_event_date": args.recommender_min_event_date,
            "recommender_max_event_date": args.recommender_max_event_date,
            "bad_ts_threshold": args.bad_ts_threshold,
            "split_version": args.split_version,
        },
        "quality": {
            "raw_total_rows": total_rows if build_shared else None,
            "raw_exact_duplicate_rows": exact_duplicate_rows if build_shared else None,
            "raw_bad_timestamp_rows": bad_rows if build_shared else None,
            "raw_bad_timestamp_ratio": round(bad_ratio, 6) if build_shared else None,
        },
        "artifacts": {},
    }
    for artifact_name, _, _, artifact_path in artifacts:
        manifest["artifacts"][artifact_name] = {
            "path": str(artifact_path),
            "rows": artifact_rows[artifact_name],
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
