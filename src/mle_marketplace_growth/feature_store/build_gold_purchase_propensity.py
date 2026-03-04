"""Build purchase-propensity gold datasets for a strict 12-month panel."""

import argparse
import calendar
import json
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb

from .build_helpers import copy_table_to_csv, load_sql_assets, load_yaml_defaults, run_dq_check
from .build_shared_silver import bootstrap_silver


def _add_month(current: date) -> date:
    year = current.year + (1 if current.month == 12 else 0)
    month = 1 if current.month == 12 else current.month + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _shift_month(current: date, delta_months: int) -> date:
    month_index = current.month - 1 + delta_months
    year = current.year + month_index // 12
    month = month_index % 12 + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _generate_snapshot_dates(panel_end_date: date) -> list[date]:
    start_date = _shift_month(panel_end_date, -11)
    snapshots = []
    current = start_date
    for _ in range(12):
        snapshots.append(current)
        current = _add_month(current)
    if snapshots[-1] != panel_end_date:
        raise ValueError("Derived monthly snapshot panel does not end on --panel-end-date")
    return snapshots


def _resolve_purchase_paths(propensity_root: Path, as_of_date: date) -> tuple[Path, Path, Path, Path]:
    as_of_partition = f"as_of_date={as_of_date.isoformat()}"
    labels_path = propensity_root / "labels" / as_of_partition / "labels.csv"
    features_path = propensity_root / "user_features_asof" / as_of_partition / "user_features_asof.csv"
    propensity_train_path = propensity_root / "propensity_train_dataset" / as_of_partition / "propensity_train_dataset.csv"
    manifest_path = propensity_root / "_meta" / as_of_partition / "run_manifest.json"
    return labels_path, features_path, propensity_train_path, manifest_path


def _build_purchase_gold(connection: duckdb.DuckDBPyConnection, sql: dict[str, str], as_of_date: date) -> None:
    for statement in [
        sql["propensity_labels"].replace("{as_of_date}", as_of_date.isoformat()),
        sql["propensity_user_features"].replace("{as_of_date}", as_of_date.isoformat()),
        sql["propensity_train_dataset"].replace("{as_of_date}", as_of_date.isoformat()),
    ]:
        connection.execute(statement)


def _purchase_dq_checks(sql: dict[str, str]) -> list[tuple[str, str]]:
    return [
        (sql["dq_user_features_duplicates"], "DQ_GOLD_001 failed: duplicate (user_id, as_of_date) in gold_user_features_asof"),
        (sql["dq_invalid_labels"], "DQ_GOLD_003 failed: invalid label metadata in gold_labels"),
        (sql["dq_propensity_train_duplicates"], "DQ_GOLD_009 failed: duplicate (user_id, as_of_date) in gold_propensity_train_dataset"),
    ]


def main() -> None:
    # Parse CLI + config defaults.
    parser = argparse.ArgumentParser(description="Build purchase-propensity gold datasets for a strict 12-month panel.")
    parser.add_argument("--config", required=True, help="Engine YAML config file")
    parser.add_argument("--shared-config", default=None, help="Optional shared YAML config file")
    parser.add_argument("--input-csv", default=None, help="Path to raw source CSV (metadata only for this command)")
    parser.add_argument("--output-root", default=None, help="Output root path for silver/gold materializations")
    parser.add_argument("--panel-end-date", default=None, help="End anchor (YYYY-MM-DD) for strict 12-month panel")
    parser.add_argument("--bad-ts-threshold", type=float, default=None, help="Ignored in this command; shared layer already enforces it")
    args = parser.parse_args()
    defaults = load_yaml_defaults(args.shared_config, "Shared config")
    defaults.update(load_yaml_defaults(args.config, "Engine config"))
    cfg = defaults.get
    args.input_csv = args.input_csv or cfg("input_csv", "data/bronze/online_retail_ii/raw.csv")
    args.output_root = args.output_root or cfg("output_root", "data")
    args.panel_end_date = args.panel_end_date or cfg("panel_end_date", None)
    args.bad_ts_threshold = float(args.bad_ts_threshold if args.bad_ts_threshold is not None else 0.01)

    # Resolve inputs and required assets.
    if not args.panel_end_date:
        raise ValueError("--panel-end-date is required")
    panel_end_date = date.fromisoformat(args.panel_end_date)

    input_csv = Path(args.input_csv)
    output_root = Path(args.output_root)
    sql = load_sql_assets(Path(__file__).resolve().parent / "sql")
    silver_path = output_root / "silver" / "transactions_line_items" / "transactions_line_items.csv"
    propensity_root = output_root / "gold" / "feature_store" / "purchase_propensity"

    # Bootstrap shared silver and validate available time bounds.
    connection = duckdb.connect(database=":memory:")
    bootstrap_silver(
        connection,
        build_shared=False,
        input_csv=input_csv,
        silver_path=silver_path,
        sql=sql,
        bad_ts_threshold=args.bad_ts_threshold,
    )
    silver_min_date, silver_max_date = connection.execute("SELECT min(event_date), max(event_date) FROM silver_transactions_line_items").fetchone()
    if silver_min_date is None or silver_max_date is None:
        raise ValueError("No rows in silver_transactions_line_items; cannot build gold layers")

    # Build and export one gold snapshot per month for the strict 12-month panel.
    for as_of_date in _generate_snapshot_dates(panel_end_date):
        if as_of_date < silver_min_date or as_of_date > silver_max_date:
            raise ValueError(
                f"Purchase propensity as_of_date {as_of_date} is outside available silver event_date bounds "
                f"[{silver_min_date}, {silver_max_date}]"
            )

        _build_purchase_gold(connection, sql, as_of_date)
        for dq_sql, dq_error in _purchase_dq_checks(sql):
            run_dq_check(connection, dq_sql, dq_error)

        labels_path, features_path, propensity_train_path, manifest_path = _resolve_purchase_paths(propensity_root, as_of_date)
        artifacts = [
            ("gold_labels", "gold_labels", labels_path),
            ("gold_user_features_asof", "gold_user_features_asof", features_path),
            ("gold_propensity_train_dataset", "gold_propensity_train_dataset", propensity_train_path),
        ]
        artifact_rows: dict[str, int] = {}
        for artifact_name, table_name, artifact_path in artifacts:
            row_count = copy_table_to_csv(connection, table_name, artifact_path, sql["copy_table_to_csv"], sql["count_rows"])
            artifact_rows[artifact_name] = row_count
            print(f"Wrote gold table: {artifact_path} ({row_count} rows)")

        # Emit per-snapshot run metadata.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "input": {"path": str(input_csv)},
            "params": {"build_engine": "purchase_propensity", "panel_end_date": panel_end_date.isoformat(), "as_of_date": as_of_date.isoformat()},
            "quality": {"raw_total_rows": None, "raw_exact_duplicate_rows": None, "raw_bad_timestamp_rows": None, "raw_bad_timestamp_ratio": None},
            "artifacts": {name: {"path": str(path), "rows": artifact_rows[name]} for name, _, path in artifacts},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
