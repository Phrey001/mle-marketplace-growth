"""Shared silver bootstrap and CLI for feature-store builds."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .build_helpers import REQUIRED_SOURCE_COLUMNS, copy_table_to_csv, load_sql_assets, load_yaml_defaults, sql_quote


def bootstrap_silver(connection: duckdb.DuckDBPyConnection, *, build_shared: bool, input_csv: Path, silver_path: Path, sql: dict[str, str], bad_ts_threshold: float) -> dict[str, float | int]:
    total_rows = bad_rows = exact_duplicate_rows = 0
    bad_ratio = 0.0

    # Shared silver build path: read raw source, run quality checks, materialize silver.
    if build_shared:
        create_raw_source_sql = sql["create_raw_source"].replace("{input_csv}", sql_quote(str(input_csv)))
        connection.execute(create_raw_source_sql)

        source_schema = connection.execute(sql["describe_raw_source"]).fetchall()
        available_columns = {row[0] for row in source_schema}
        missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - available_columns)
        if missing_columns:
            raise ValueError(f"Missing required source columns: {missing_columns}")

        timestamp_quality = connection.execute(sql["raw_bad_timestamp_counts"]).fetchone()
        total_rows = int(timestamp_quality[0] or 0)
        bad_rows = int(timestamp_quality[1] or 0)
        exact_duplicate_rows = int(connection.execute(sql["raw_exact_duplicate_rows"]).fetchone()[0] or 0)
        bad_ratio = (bad_rows / total_rows) if total_rows else 0.0
        if bad_ratio > bad_ts_threshold:
            raise ValueError(f"Timestamp parse failure ratio {bad_ratio:.4f} exceeded threshold {bad_ts_threshold:.4f}")
        connection.execute(sql["silver_transactions_line_items"])
        return {
            "raw_total_rows": total_rows,
            "raw_exact_duplicate_rows": exact_duplicate_rows,
            "raw_bad_timestamp_rows": bad_rows,
            "raw_bad_timestamp_ratio": bad_ratio,
        }

    # Reuse path: load prebuilt silver CSV into DuckDB and normalize schema.
    if not silver_path.exists():
        raise FileNotFoundError(f"Shared silver CSV not found: {silver_path}. Run feature_store.build_shared_silver first.")
    create_silver_from_csv_sql = sql["create_silver_from_shared_csv"].replace("{silver_csv}", sql_quote(str(silver_path)))
    connection.execute(create_silver_from_csv_sql)
    connection.execute(sql["silver_transactions_line_items"])
    return {"raw_total_rows": 0, "raw_exact_duplicate_rows": 0, "raw_bad_timestamp_rows": 0, "raw_bad_timestamp_ratio": 0.0}


def main() -> None:
    # Parse CLI + config defaults.
    parser = argparse.ArgumentParser(description="Build shared silver feature-store layer.")
    parser.add_argument("--shared-config", required=True, help="Shared YAML config file")
    parser.add_argument("--input-csv", default=None, help="Path to raw Online Retail II CSV")
    parser.add_argument("--output-root", default=None, help="Output root path for silver/gold materializations")
    parser.add_argument("--bad-ts-threshold", type=float, default=None, help="Maximum tolerated ratio of rows with unparsable InvoiceDate before failing")
    args = parser.parse_args()
    cfg = load_yaml_defaults(args.shared_config, "Shared config").get
    args.input_csv = args.input_csv or cfg("input_csv", "data/bronze/online_retail_ii/raw.csv")
    args.output_root = args.output_root or cfg("output_root", "data")
    args.bad_ts_threshold = float(args.bad_ts_threshold if args.bad_ts_threshold is not None else 0.01)

    # Resolve paths + load SQL assets.
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    output_root = Path(args.output_root)
    silver_path = output_root / "silver" / "transactions_line_items" / "transactions_line_items.csv"
    gold_root = output_root / "gold" / "feature_store"
    manifest_path = gold_root / "_meta" / "run_manifest.json"
    sql = load_sql_assets(Path(__file__).resolve().parent / "sql")

    # Execute shared silver build.
    connection = duckdb.connect(database=":memory:")
    quality = bootstrap_silver(
        connection,
        build_shared=True,
        input_csv=input_csv,
        silver_path=silver_path,
        sql=sql,
        bad_ts_threshold=args.bad_ts_threshold,
    )
    row_count = copy_table_to_csv(connection, "silver_transactions_line_items", silver_path, sql["copy_table_to_csv"], sql["count_rows"])
    print(f"Wrote silver table: {silver_path} ({row_count} rows)")

    # Emit run metadata.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input": {"path": str(input_csv)},
        "params": {"build_engine": "shared", "bad_ts_threshold": args.bad_ts_threshold},
        "quality": {
            "raw_total_rows": quality["raw_total_rows"],
            "raw_exact_duplicate_rows": quality["raw_exact_duplicate_rows"],
            "raw_bad_timestamp_rows": quality["raw_bad_timestamp_rows"],
            "raw_bad_timestamp_ratio": round(float(quality["raw_bad_timestamp_ratio"]), 6),
        },
        "artifacts": {"silver_transactions_line_items": {"path": str(silver_path), "rows": row_count}},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
