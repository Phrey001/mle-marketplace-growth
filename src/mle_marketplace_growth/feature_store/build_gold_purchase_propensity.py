"""Build purchase-propensity gold datasets for a strict 12-month panel."""

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
from mle_marketplace_growth.helpers import cfg_required, generate_snapshot_dates, load_yaml_defaults, write_json

from .build_helpers import copy_table_to_parquet, load_shared_silver_table, load_sql_assets, run_dq_check


def _resolve_purchase_paths(propensity_root: Path, as_of_date: date) -> tuple[Path, Path, Path]:
    """Return output parquet paths for a given as_of_date partition."""
    as_of_partition = f"as_of_date={as_of_date.isoformat()}"
    labels_path = propensity_root / "labels" / as_of_partition / "labels.parquet"
    features_path = propensity_root / "user_features_asof" / as_of_partition / "user_features_asof.parquet"
    propensity_train_path = propensity_root / "propensity_train_dataset" / as_of_partition / "propensity_train_dataset.parquet"
    return labels_path, features_path, propensity_train_path


def _build_purchase_gold(connection: duckdb.DuckDBPyConnection, sql: dict[str, str], as_of_date: date) -> None:
    """Materialize gold labels/features/train dataset for a single snapshot date."""
    for statement in [
        sql["propensity_labels"].replace("{as_of_date}", as_of_date.isoformat()),
        sql["propensity_user_features"].replace("{as_of_date}", as_of_date.isoformat()),
        sql["propensity_train_dataset"].replace("{as_of_date}", as_of_date.isoformat()),
    ]:
        connection.execute(statement)


def _purchase_dq_checks(sql: dict[str, str]) -> list[tuple[str, str]]:
    """Return DQ SQL checks paired with their failure messages."""
    return [
        (sql["dq_user_features_duplicates"], "DQ_GOLD_001 failed: duplicate (user_id, as_of_date) in gold_user_features_asof"),
        (sql["dq_invalid_labels"], "DQ_GOLD_003 failed: invalid label metadata in gold_labels"),
        (sql["dq_propensity_train_duplicates"], "DQ_GOLD_009 failed: duplicate (user_id, as_of_date) in gold_propensity_train_dataset"),
    ]


def main() -> None:
    """CLI entrypoint for building purchase-propensity gold datasets."""
    # Parse CLI + config defaults.
    parser = argparse.ArgumentParser(description="Build purchase-propensity gold datasets for a strict 12-month panel.")
    parser.add_argument("--config", required=True, help="Engine YAML config file")
    args = parser.parse_args()
    cfg = load_yaml_defaults(args.config, "Engine config")

    # Resolve inputs and required assets.
    panel_end_date = date.fromisoformat(str(cfg_required(cfg, "panel_end_date")))

    output_root = Path("data")
    sql = load_sql_assets(Path(__file__).resolve().parent / "sql")
    silver_path = output_root / "silver" / "transactions_line_items" / "transactions_line_items.parquet"
    propensity_root = output_root / "gold" / "feature_store" / "purchase_propensity"
    manifest_path = propensity_root / "_meta" / f"panel_end_date={panel_end_date.isoformat()}" / "run_manifest.json"

    # Load canonical shared silver and validate available time bounds.
    connection = duckdb.connect(database=":memory:")
    load_shared_silver_table(connection, silver_path)
    silver_min_date, silver_max_date = connection.execute("SELECT min(event_date), max(event_date) FROM silver_transactions_line_items").fetchone()
    if silver_min_date is None or silver_max_date is None:
        raise ValueError(
            "Invalid upstream data: silver_transactions_line_items has no usable event_date values, "
            "so gold layers cannot be built."
        )

    # Build and export one gold snapshot per month for the strict 12-month panel.
    snapshot_manifests: list[dict] = []
    for as_of_date in generate_snapshot_dates(panel_end_date):
        if as_of_date < silver_min_date or as_of_date > silver_max_date:
            raise ValueError(
                f"Purchase propensity as_of_date {as_of_date} is outside available silver event_date bounds "
                f"[{silver_min_date}, {silver_max_date}]"
            )

        # Build gold tables for the snapshot and enforce DQ checks before export.
        _build_purchase_gold(connection, sql, as_of_date)
        for dq_sql, dq_error in _purchase_dq_checks(sql):
            run_dq_check(connection, dq_sql, dq_error)

        labels_path, features_path, propensity_train_path = _resolve_purchase_paths(propensity_root, as_of_date)
        artifacts = [
            ("gold_labels", "gold_labels", labels_path),
            ("gold_user_features_asof", "gold_user_features_asof", features_path),
            ("gold_propensity_train_dataset", "gold_propensity_train_dataset", propensity_train_path),
        ]
        artifact_rows: dict[str, int] = {}
        # Persist each gold table to parquet with row-count bookkeeping for manifests.
        for artifact_name, table_name, artifact_path in artifacts:
            row_count = copy_table_to_parquet(connection, table_name, artifact_path, sql["count_rows"])
            artifact_rows[artifact_name] = row_count
            print(f"Wrote gold table: {artifact_path} ({row_count} rows)")

        snapshot_manifests.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "artifacts": {name: {"path": str(path), "rows": artifact_rows[name]} for name, _, path in artifacts},
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input": {"silver_transactions_line_items_path": str(silver_path)},
        "params": {"build_engine": "purchase_propensity", "panel_end_date": panel_end_date.isoformat()},
        "quality": {"raw_total_rows": None, "raw_bad_timestamp_rows": None, "raw_bad_timestamp_ratio": None},
        "snapshots": snapshot_manifests,
    }
    write_json(manifest_path, manifest)
    print(f"Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
