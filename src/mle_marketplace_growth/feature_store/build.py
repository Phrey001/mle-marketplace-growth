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


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _copy_table_to_csv(connection: duckdb.DuckDBPyConnection, table_name: str, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quoted_path = _sql_quote(str(output_path))
    connection.execute(
        f"""
        COPY (
          SELECT * FROM {table_name}
        ) TO '{quoted_path}' (HEADER, DELIMITER ',');
        """
    )
    row_count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return int(row_count)


def main() -> None:
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

    sql_dir = Path(__file__).resolve().parent / "sql"
    silver_sql = (sql_dir / "silver_transactions_line_items.sql").read_text(encoding="utf-8")
    interactions_sql = (sql_dir / "gold_interaction_events.sql").read_text(encoding="utf-8")
    user_item_splits_sql_template = (sql_dir / "gold_user_item_splits.sql").read_text(encoding="utf-8")
    labels_sql_template = (sql_dir / "gold_labels.sql").read_text(encoding="utf-8")
    user_features_sql_template = (sql_dir / "gold_user_features_asof.sql").read_text(encoding="utf-8")
    uplift_train_sql_template = (sql_dir / "gold_uplift_train_dataset.sql").read_text(encoding="utf-8")

    silver_path = output_root / "silver" / "transactions_line_items" / "transactions_line_items.csv"
    interactions_path = output_root / "gold" / "feature_store" / "interaction_events" / "interaction_events.csv"
    user_item_splits_path = output_root / "gold" / "feature_store" / "user_item_splits" / "user_item_splits.csv"

    connection = duckdb.connect(database=":memory:")
    quoted_input_csv = _sql_quote(str(input_csv))
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW raw_source AS
        SELECT *
        FROM read_csv_auto('{quoted_input_csv}', header=true, sample_size=-1);
        """
    )

    source_schema = connection.execute("DESCRIBE SELECT * FROM raw_source").fetchall()
    available_columns = {row[0] for row in source_schema}
    missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - available_columns)
    if missing_columns:
        raise ValueError(f"Missing required source columns: {missing_columns}")

    timestamp_quality = connection.execute(
        """
        SELECT
          COUNT(*) AS total_rows,
          SUM(
            CASE
              WHEN try_strptime(trim(CAST("InvoiceDate" AS VARCHAR)), '%Y-%m-%d %H:%M:%S') IS NULL
              THEN 1 ELSE 0
            END
          ) AS bad_timestamp_rows
        FROM raw_source
        """
    ).fetchone()
    total_rows = int(timestamp_quality[0] or 0)
    bad_rows = int(timestamp_quality[1] or 0)
    bad_ratio = (bad_rows / total_rows) if total_rows else 0.0
    if bad_ratio > args.bad_ts_threshold:
        raise ValueError(
            f"Timestamp parse failure ratio {bad_ratio:.4f} exceeded threshold {args.bad_ts_threshold:.4f}"
        )

    connection.execute(silver_sql)

    if args.as_of_date:
        as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
    else:
        as_of_value = connection.execute("SELECT MAX(event_date) FROM silver_transactions_line_items").fetchone()[0]
        if as_of_value is None:
            raise ValueError("No rows in silver_transactions_line_items; cannot resolve as_of_date")
        as_of_date = as_of_value
    labels_path = (
        output_root
        / "gold"
        / "feature_store"
        / "labels"
        / f"as_of_date={as_of_date.isoformat()}"
        / "labels.csv"
    )
    uplift_train_path = (
        output_root
        / "gold"
        / "feature_store"
        / "uplift_train_dataset"
        / f"as_of_date={as_of_date.isoformat()}"
        / "uplift_train_dataset.csv"
    )
    manifest_path = (
        output_root
        / "gold"
        / "feature_store"
        / "_meta"
        / f"as_of_date={as_of_date.isoformat()}"
        / "run_manifest.json"
    )

    user_features_sql = user_features_sql_template.replace("{as_of_date}", as_of_date.isoformat())
    labels_sql = labels_sql_template.replace("{as_of_date}", as_of_date.isoformat())
    uplift_train_sql = uplift_train_sql_template.replace("{as_of_date}", as_of_date.isoformat())
    user_item_splits_sql = user_item_splits_sql_template.replace("{split_version}", args.split_version)
    connection.execute(interactions_sql)
    connection.execute(user_item_splits_sql)
    connection.execute(labels_sql)
    connection.execute(user_features_sql)
    connection.execute(uplift_train_sql)

    invalid_split_count = connection.execute(
        """
        SELECT COUNT(*) FROM gold_user_item_splits
        WHERE split NOT IN ('train', 'val', 'test')
           OR split_version = ''
        """
    ).fetchone()[0]
    if int(invalid_split_count) > 0:
        raise ValueError("DQ_GOLD_005 failed: invalid split value in gold_user_item_splits")

    chronology_violation_count = connection.execute(
        """
        WITH split_times AS (
          SELECT
            user_id,
            max(CASE WHEN split = 'train' THEN try_strptime(event_ts, '%Y-%m-%d %H:%M:%S') END) AS max_train_ts,
            min(CASE WHEN split = 'val' THEN try_strptime(event_ts, '%Y-%m-%d %H:%M:%S') END) AS min_val_ts,
            min(CASE WHEN split = 'test' THEN try_strptime(event_ts, '%Y-%m-%d %H:%M:%S') END) AS min_test_ts
          FROM gold_user_item_splits
          GROUP BY user_id
        )
        SELECT COUNT(*) FROM split_times
        WHERE (max_train_ts IS NOT NULL AND min_val_ts IS NOT NULL AND max_train_ts > min_val_ts)
           OR (min_val_ts IS NOT NULL AND min_test_ts IS NOT NULL AND min_val_ts > min_test_ts)
        """
    ).fetchone()[0]
    if int(chronology_violation_count) > 0:
        raise ValueError("DQ_GOLD_006 failed: chronology violation in gold_user_item_splits")

    duplicate_grain_count = connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT user_id, as_of_date, COUNT(*) AS c
          FROM gold_user_features_asof
          GROUP BY user_id, as_of_date
          HAVING c > 1
        )
        """
    ).fetchone()[0]
    if int(duplicate_grain_count) > 0:
        raise ValueError("DQ_GOLD_001 failed: duplicate (user_id, as_of_date) in gold_user_features_asof")

    invalid_label_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM gold_labels
        WHERE label_name NOT IN ('net_revenue_30d', 'purchase_30d')
           OR window_days <> 30
        """
    ).fetchone()[0]
    if int(invalid_label_rows) > 0:
        raise ValueError("DQ_GOLD_003 failed: invalid label metadata in gold_labels")

    train_dataset_duplicate_count = connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT user_id, as_of_date, COUNT(*) AS c
          FROM gold_uplift_train_dataset
          GROUP BY user_id, as_of_date
          HAVING c > 1
        )
        """
    ).fetchone()[0]
    if int(train_dataset_duplicate_count) > 0:
        raise ValueError("DQ_GOLD_009 failed: duplicate (user_id, as_of_date) in gold_uplift_train_dataset")

    features_path = (
        output_root
        / "gold"
        / "feature_store"
        / "user_features_asof"
        / f"as_of_date={as_of_date.isoformat()}"
        / "user_features_asof.csv"
    )

    silver_count = _copy_table_to_csv(connection, "silver_transactions_line_items", silver_path)
    interactions_count = _copy_table_to_csv(connection, "gold_interaction_events", interactions_path)
    splits_count = _copy_table_to_csv(connection, "gold_user_item_splits", user_item_splits_path)
    labels_count = _copy_table_to_csv(connection, "gold_labels", labels_path)
    features_count = _copy_table_to_csv(connection, "gold_user_features_asof", features_path)
    uplift_train_count = _copy_table_to_csv(connection, "gold_uplift_train_dataset", uplift_train_path)

    print(f"Wrote silver table: {silver_path} ({silver_count} rows)")
    print(f"Wrote gold table: {interactions_path} ({interactions_count} rows)")
    print(f"Wrote gold table: {user_item_splits_path} ({splits_count} rows)")
    print(f"Wrote gold table: {labels_path} ({labels_count} rows)")
    print(f"Wrote gold table: {features_path} ({features_count} rows)")
    print(f"Wrote gold table: {uplift_train_path} ({uplift_train_count} rows)")

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
            "raw_bad_timestamp_rows": bad_rows,
            "raw_bad_timestamp_ratio": round(bad_ratio, 6),
        },
        "artifacts": {
            "silver_transactions_line_items": {"path": str(silver_path), "rows": silver_count},
            "gold_interaction_events": {"path": str(interactions_path), "rows": interactions_count},
            "gold_user_item_splits": {"path": str(user_item_splits_path), "rows": splits_count},
            "gold_labels": {"path": str(labels_path), "rows": labels_count},
            "gold_user_features_asof": {"path": str(features_path), "rows": features_count},
            "gold_uplift_train_dataset": {"path": str(uplift_train_path), "rows": uplift_train_count},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
