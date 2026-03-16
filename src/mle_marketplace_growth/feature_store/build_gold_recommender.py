"""Recommender gold-layer build steps and CLI."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from mle_marketplace_growth.helpers import cfg_required, load_yaml_defaults, write_json

from .build_helpers import copy_table_to_parquet, load_shared_silver_table, load_sql_assets, run_dq_check


def _validate_runtime_event_date_config(raw_min_event_date: str, raw_max_event_date: str, silver_min_date, silver_max_date) -> None:
    """What: Validate recommender event-date config values against available silver dates.
    Why: Fails fast when the configured recommender window falls outside the
    silver dataset's actual date range, before rendering SQL and materializing
    gold tables.
    """
    def _parse_iso_date(raw_value: str, arg_name: str):
        """Parse one YYYY-MM-DD string into a date for bounds validation."""
        try:
            return datetime.strptime(raw_value, "%Y-%m-%d").date()
        except ValueError as error:
            raise ValueError(f"{arg_name} must be YYYY-MM-DD: {raw_value}") from error

    recommender_min_date = _parse_iso_date(raw_min_event_date, "recommender_min_event_date")
    recommender_max_date = _parse_iso_date(raw_max_event_date, "recommender_max_event_date")
    if (
        recommender_min_date > recommender_max_date
        or recommender_min_date < silver_min_date
        or recommender_max_date > silver_max_date
    ):
        raise ValueError(
            "Invalid runtime config: recommender event-date window "
            f"[{recommender_min_date}, {recommender_max_date}] must satisfy "
            f"{silver_min_date} <= recommender_min_event_date <= recommender_max_event_date <= {silver_max_date}."
        )


def build_recommender_gold(
    connection: duckdb.DuckDBPyConnection,
    sql: dict[str, str],
    *,
    recommender_min_event_date: str | None,
    recommender_max_event_date: str | None,
    silver_min_date,
    silver_max_date,
) -> None:
    """What: Materialize recommender gold tables in DuckDB.
    Why: Builds the full recommender gold contract from canonical shared silver.
    """
    # Validate that the configured recommender min/max dates are inside the
    # dataset bounds discovered from silver, before rendering the SQL template.
    _validate_runtime_event_date_config(
        recommender_min_event_date,
        recommender_max_event_date,
        silver_min_date,
        silver_max_date,
    )
    # interaction_events.sql is the only recommender gold SQL template that needs
    # runtime config injection, and Python only supplies the min/max date values.
    interaction_events_sql = (
        sql["recommender_interactions"]
        .replace("{recommender_min_event_date}", recommender_min_event_date)
        .replace("{recommender_max_event_date}", recommender_max_event_date)
    )
    for statement in [interaction_events_sql, sql["recommender_user_item_splits"], sql["recommender_user_index"], sql["recommender_item_index"]]:
        connection.execute(statement)


def recommender_dq_checks(sql: dict[str, str]) -> list[tuple[str, str]]:
    """What: Return recommender gold DQ checks and failure messages.
    Why: Keeps split-validity and chronology enforcement explicit in one place.
    """
    return [
        (sql["dq_invalid_split"], "DQ_GOLD_005 failed: invalid split value in gold_user_item_splits"),
        (sql["dq_split_chronology"], "DQ_GOLD_006 failed: chronology violation in gold_user_item_splits"),
    ]


def main() -> None:
    """What: CLI entrypoint for recommender gold-layer build.
    Why: Loads config, builds recommender gold artifacts, runs DQ, and writes manifest metadata.
    """
    # Parse CLI + config defaults.
    parser = argparse.ArgumentParser(description="Build recommender gold feature-store layer.")
    parser.add_argument("--config", required=True, help="Engine YAML config file")
    args = parser.parse_args()
    defaults = load_yaml_defaults(args.config, "Engine config")
    recommender_min_event_date = str(cfg_required(defaults, "recommender_min_event_date"))
    recommender_max_event_date = str(cfg_required(defaults, "recommender_max_event_date"))

    # Resolve paths + load SQL assets.
    output_root = Path("data")
    sql = load_sql_assets(Path(__file__).resolve().parent / "sql")
    silver_path = output_root / "silver" / "transactions_line_items" / "transactions_line_items.parquet"
    gold_root = output_root / "gold" / "feature_store"
    recommender_root = gold_root / "recommender"
    manifest_path = recommender_root / "_meta" / "run_manifest.json"

    # Load canonical shared silver, then discover the dataset's actual available
    # event-date bounds from the loaded silver table.
    connection = duckdb.connect(database=":memory:")
    load_shared_silver_table(connection, silver_path)
    silver_min_date, silver_max_date = connection.execute("SELECT min(event_date), max(event_date) FROM silver_transactions_line_items").fetchone()
    if silver_min_date is None or silver_max_date is None:
        raise ValueError(
            "Invalid upstream data: silver_transactions_line_items has no usable event_date values, "
            "so gold layers cannot be built."
        )

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
        "input": {"silver_transactions_line_items_path": str(silver_path)},
        "params": {
            "build_engine": "recommender",
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
