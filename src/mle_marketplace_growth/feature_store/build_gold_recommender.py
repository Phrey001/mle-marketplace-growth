"""Recommender gold-layer build steps and CLI."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from mle_marketplace_growth.helpers import load_yaml_defaults, write_json

from .build_helpers import copy_table_to_parquet, load_shared_silver_table, load_sql_assets, run_dq_check


def _parse_iso_date(raw_value: str, arg_name: str):
    try:
        return datetime.strptime(raw_value, "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError(f"{arg_name} must be YYYY-MM-DD: {raw_value}") from error


def _build_recommender_filters(raw_min_event_date: str | None, raw_max_event_date: str | None, silver_min_date, silver_max_date) -> list[str]:
    recommender_time_filters = []
    recommender_min_date = silver_min_date
    recommender_max_date = silver_max_date
    if raw_min_event_date:
        recommender_min_date = _parse_iso_date(raw_min_event_date, "--recommender-min-event-date")
        recommender_time_filters.append(f"AND event_date >= CAST('{raw_min_event_date}' AS DATE)")
    if raw_max_event_date:
        recommender_max_date = _parse_iso_date(raw_max_event_date, "--recommender-max-event-date")
        recommender_time_filters.append(f"AND event_date <= CAST('{raw_max_event_date}' AS DATE)")
    if recommender_min_date < silver_min_date or recommender_max_date > silver_max_date:
        raise ValueError(
            f"Recommender event-date bounds [{recommender_min_date}, {recommender_max_date}] exceed available silver bounds "
            f"[{silver_min_date}, {silver_max_date}]"
        )
    return recommender_time_filters


def build_recommender_gold(
    connection: duckdb.DuckDBPyConnection,
    sql: dict[str, str],
    *,
    recommender_min_event_date: str | None,
    recommender_max_event_date: str | None,
    silver_min_date,
    silver_max_date,
) -> None:
    recommender_time_filters = _build_recommender_filters(
        recommender_min_event_date,
        recommender_max_event_date,
        silver_min_date,
        silver_max_date,
    )
    interactions_sql = sql["recommender_interactions"].replace(
        "{recommender_time_filters}",
        "\n  ".join(recommender_time_filters) if recommender_time_filters else "",
    )
    for statement in [interactions_sql, sql["recommender_user_item_splits"], sql["recommender_user_index"], sql["recommender_item_index"]]:
        connection.execute(statement)


def recommender_dq_checks(sql: dict[str, str]) -> list[tuple[str, str]]:
    return [
        (sql["dq_invalid_split"], "DQ_GOLD_005 failed: invalid split value in gold_user_item_splits"),
        (sql["dq_split_chronology"], "DQ_GOLD_006 failed: chronology violation in gold_user_item_splits"),
    ]


def main() -> None:
    # Parse CLI + config defaults.
    parser = argparse.ArgumentParser(description="Build recommender gold feature-store layer.")
    parser.add_argument("--config", required=True, help="Engine YAML config file")
    parser.add_argument("--shared-config", default=None, help="Optional shared YAML config file")
    parser.add_argument("--output-root", default=None, help="Output root path for silver/gold materializations")
    parser.add_argument("--recommender-min-event-date", default=None, help="Optional lower bound event_date filter for recommender datasets in YYYY-MM-DD")
    parser.add_argument("--recommender-max-event-date", default=None, help="Optional upper bound event_date filter for recommender datasets in YYYY-MM-DD")
    args = parser.parse_args()
    defaults = load_yaml_defaults(args.shared_config, "Shared config")
    defaults.update(load_yaml_defaults(args.config, "Engine config"))
    output_root_value = str(args.output_root or defaults.get("output_root", "data"))
    recommender_min_event_date = args.recommender_min_event_date or defaults.get("recommender_min_event_date", None)
    recommender_max_event_date = args.recommender_max_event_date or defaults.get("recommender_max_event_date", None)

    # Resolve paths + load SQL assets.
    output_root = Path(output_root_value)
    sql = load_sql_assets(Path(__file__).resolve().parent / "sql")
    shared_db_path = output_root / "_tmp" / "feature_store.duckdb"
    gold_root = output_root / "gold" / "feature_store"
    recommender_root = gold_root / "recommender"
    manifest_path = gold_root / "_meta" / "run_manifest.json"

    # Load canonical shared silver and validate available time bounds.
    connection = duckdb.connect(database=":memory:")
    load_shared_silver_table(connection, shared_db_path)
    silver_min_date, silver_max_date = connection.execute("SELECT min(event_date), max(event_date) FROM silver_transactions_line_items").fetchone()
    if silver_min_date is None or silver_max_date is None:
        raise ValueError("No rows in silver_transactions_line_items; cannot build gold layers")

    # Build recommender gold tables + run DQ checks.
    build_recommender_gold(
        connection,
        sql,
        recommender_min_event_date=recommender_min_event_date,
        recommender_max_event_date=recommender_max_event_date,
        silver_min_date=silver_min_date,
        silver_max_date=silver_max_date,
    )
    for dq_sql, dq_error in recommender_dq_checks(sql):
        run_dq_check(connection, dq_sql, dq_error)

    artifacts = [
        ("gold_interaction_events", "gold_interaction_events", recommender_root / "interaction_events" / "interaction_events.parquet"),
        ("gold_user_item_splits", "gold_user_item_splits", recommender_root / "user_item_splits" / "user_item_splits.parquet"),
        ("gold_recommender_user_index", "gold_recommender_user_index", recommender_root / "user_index" / "user_index.parquet"),
        ("gold_recommender_item_index", "gold_recommender_item_index", recommender_root / "item_index" / "item_index.parquet"),
    ]
    artifact_rows: dict[str, int] = {}
    for artifact_name, table_name, artifact_path in artifacts:
        row_count = copy_table_to_parquet(connection, table_name, artifact_path, sql["count_rows"])
        artifact_rows[artifact_name] = row_count
        print(f"Wrote gold table: {artifact_path} ({row_count} rows)")

    # Emit run metadata.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input": {"shared_db_path": str(shared_db_path)},
        "params": {
            "build_engine": "recommender",
            "split_version": "time_rank_v1",
            "recommender_min_event_date": recommender_min_event_date,
            "recommender_max_event_date": recommender_max_event_date,
        },
        "quality": {"raw_total_rows": None, "raw_bad_timestamp_rows": None, "raw_bad_timestamp_ratio": None},
        "artifacts": {name: {"path": str(path), "rows": artifact_rows[name]} for name, _, path in artifacts},
    }
    write_json(manifest_path, manifest)
    print(f"Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
