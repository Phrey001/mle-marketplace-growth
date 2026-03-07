"""Build the shared silver layer from raw source data."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .build_helpers import copy_table_to_parquet, load_sql_assets, load_yaml_defaults

BAD_TS_THRESHOLD = 0.01


def _sql_quote(value: str) -> str:
    """Escape single quotes in SQL path/file literals (not row content)."""
    return value.replace("'", "''")


def bootstrap_silver(
    connection: duckdb.DuckDBPyConnection,
    *,
    input_csv: Path,
    sql: dict[str, str],
) -> dict[str, float | int]:
    """Load raw data, run DQ checks, and materialize canonical silver table."""
    required_source_columns = {"Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"}
    create_raw_source_sql = sql["create_raw_source"].replace("{input_csv}", _sql_quote(str(input_csv)))
    connection.execute(create_raw_source_sql)

    # DQ 1: fail fast if source schema misses required columns.
    available_columns = {row[0] for row in connection.execute(sql["describe_raw_source"]).fetchall()}
    missing_columns = sorted(required_source_columns - available_columns)
    if missing_columns:
        raise ValueError(f"Missing required source columns: {missing_columns}")

    # DQ 2: fail fast when unparsable timestamp ratio exceeds threshold.
    total_rows, bad_rows = connection.execute(sql["raw_bad_timestamp_counts"]).fetchone()
    total_rows = int(total_rows or 0)
    bad_rows = int(bad_rows or 0)
    bad_ratio = (bad_rows / total_rows) if total_rows else 0.0
    if bad_ratio > BAD_TS_THRESHOLD:
        raise ValueError(f"Timestamp parse failure ratio {bad_ratio:.4f} exceeded threshold {BAD_TS_THRESHOLD:.4f}")

    connection.execute(sql["silver_transactions_line_items"])

    # DQ 3: enforce canonical silver grain uniqueness (blocking).
    silver_duplicate_grain_rows = int(connection.execute(sql["silver_duplicate_grain_rows"]).fetchone()[0] or 0)
    if silver_duplicate_grain_rows > 0:
        raise ValueError(
            "DQ_SILVER_001 failed: duplicate rows at silver grain "
            "(invoice_id, item_id, event_ts) in silver_transactions_line_items"
        )
    return {
        "raw_total_rows": total_rows,
        "raw_bad_timestamp_rows": bad_rows,
        "raw_bad_timestamp_ratio": bad_ratio,
        "silver_duplicate_grain_rows": silver_duplicate_grain_rows,
    }


def main() -> None:
    """Parse args/config, build shared silver, and emit run manifest."""
    parser = argparse.ArgumentParser(description="Build shared silver feature-store layer.")
    parser.add_argument("--shared-config", required=True, help="Shared YAML config file")
    args = parser.parse_args()

    cfg = load_yaml_defaults(args.shared_config, "Shared config").get
    args.input_csv = cfg("input_csv", "data/bronze/online_retail_ii/raw.csv")
    args.output_root = cfg("output_root", "data")

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    output_root = Path(args.output_root)
    silver_path = output_root / "silver" / "transactions_line_items" / "transactions_line_items.parquet"
    shared_db_path = output_root / "_tmp" / "feature_store.duckdb"
    manifest_path = output_root / "silver" / "_meta" / "run_manifest.json"
    sql = load_sql_assets(Path(__file__).resolve().parent / "sql")

    shared_db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(database=str(shared_db_path))
    quality = bootstrap_silver(connection, input_csv=input_csv, sql=sql)
    row_count = copy_table_to_parquet(connection, "silver_transactions_line_items", silver_path, sql["count_rows"])
    print(f"Wrote silver table: {silver_path} ({row_count} rows)")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input": {"path": str(input_csv)},
        "params": {"build_engine": "shared", "bad_ts_threshold": BAD_TS_THRESHOLD},
        "quality": {
            "raw_total_rows": quality["raw_total_rows"],
            "raw_bad_timestamp_rows": quality["raw_bad_timestamp_rows"],
            "raw_bad_timestamp_ratio": round(float(quality["raw_bad_timestamp_ratio"]), 6),
            "silver_duplicate_grain_rows": quality["silver_duplicate_grain_rows"],
        },
        "artifacts": {
            "silver_transactions_line_items": {"path": str(silver_path), "rows": row_count},
            "shared_db": {"path": str(shared_db_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
