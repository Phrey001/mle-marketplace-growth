"""Common helpers for feature-store build orchestration."""

from datetime import datetime
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
def load_sql(sql_path: Path) -> str:
    return sql_path.read_text(encoding="utf-8")


def sql_quote(value: str) -> str:
    return value.replace("'", "''")


def copy_table_to_csv(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    output_path: Path,
    copy_sql_template: str,
    count_rows_sql_template: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quoted_path = sql_quote(str(output_path))
    copy_sql = copy_sql_template.replace("{table_name}", table_name).replace("{output_path}", quoted_path)
    connection.execute(copy_sql)
    count_rows_sql = count_rows_sql_template.replace("{table_name}", table_name)
    row_count = connection.execute(count_rows_sql).fetchone()[0]
    return int(row_count)


def run_dq_check(connection: duckdb.DuckDBPyConnection, sql: str, error_message: str) -> None:
    failing_rows = int(connection.execute(sql).fetchone()[0] or 0)
    if failing_rows > 0:
        raise ValueError(error_message)


def parse_iso_date(raw_value: str, arg_name: str):
    try:
        return datetime.strptime(raw_value, "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError(f"{arg_name} must be YYYY-MM-DD: {raw_value}") from error


def build_recommender_filters(
    raw_min_event_date: str | None, raw_max_event_date: str | None, silver_min_date, silver_max_date
) -> list[str]:
    recommender_time_filters = []
    recommender_min_date = silver_min_date
    recommender_max_date = silver_max_date
    if raw_min_event_date:
        recommender_min_date = parse_iso_date(raw_min_event_date, "--recommender-min-event-date")
        recommender_time_filters.append(f"AND event_date >= CAST('{raw_min_event_date}' AS DATE)")
    if raw_max_event_date:
        recommender_max_date = parse_iso_date(raw_max_event_date, "--recommender-max-event-date")
        recommender_time_filters.append(f"AND event_date <= CAST('{raw_max_event_date}' AS DATE)")
    if recommender_min_date < silver_min_date or recommender_max_date > silver_max_date:
        raise ValueError(
            f"Recommender event-date bounds [{recommender_min_date}, {recommender_max_date}] exceed available silver bounds "
            f"[{silver_min_date}, {silver_max_date}]"
        )
    return recommender_time_filters


def load_sql_assets(sql_dir: Path) -> dict[str, str]:
    return {
        # Pipeline SQL
        "create_raw_source": load_sql(sql_dir / "pipeline" / "create_raw_source.sql"),
        "raw_bad_timestamp_counts": load_sql(sql_dir / "pipeline" / "raw_bad_timestamp_counts.sql"),
        "raw_exact_duplicate_rows": load_sql(sql_dir / "pipeline" / "raw_exact_duplicate_rows_count.sql"),
        "copy_table_to_csv": load_sql(sql_dir / "pipeline" / "copy_table_to_csv.sql"),
        "count_rows": load_sql(sql_dir / "pipeline" / "count_rows.sql"),
        "describe_raw_source": load_sql(sql_dir / "pipeline" / "describe_raw_source.sql"),
        "max_silver_event_date": load_sql(sql_dir / "pipeline" / "max_silver_event_date.sql"),
        "create_silver_from_shared_csv": load_sql(sql_dir / "pipeline" / "create_silver_from_shared_csv.sql"),
        # Model SQL (silver + gold)
        "silver_transactions_line_items": load_sql(sql_dir / "silver" / "transactions_line_items.sql"),
        "recommender_interactions": load_sql(sql_dir / "gold" / "recommender" / "interaction_events.sql"),
        "recommender_user_item_splits": load_sql(sql_dir / "gold" / "recommender" / "user_item_splits.sql"),
        "recommender_user_index": load_sql(sql_dir / "gold" / "recommender" / "user_index.sql"),
        "recommender_item_index": load_sql(sql_dir / "gold" / "recommender" / "item_index.sql"),
        "propensity_labels": load_sql(sql_dir / "gold" / "purchase_propensity" / "labels.sql"),
        "propensity_user_features": load_sql(sql_dir / "gold" / "purchase_propensity" / "user_features_asof.sql"),
        "propensity_train_dataset": load_sql(sql_dir / "gold" / "purchase_propensity" / "propensity_train_dataset.sql"),
        # DQ SQL
        "dq_invalid_split": load_sql(sql_dir / "dq" / "gold_invalid_split_count.sql"),
        "dq_split_chronology": load_sql(sql_dir / "dq" / "gold_split_chronology_violation_count.sql"),
        "dq_user_features_duplicates": load_sql(sql_dir / "dq" / "gold_user_features_duplicate_grain_count.sql"),
        "dq_invalid_labels": load_sql(sql_dir / "dq" / "gold_invalid_label_rows_count.sql"),
        "dq_propensity_train_duplicates": load_sql(sql_dir / "dq" / "gold_propensity_train_duplicate_grain_count.sql"),
    }


def load_yaml_defaults(path_value: str | None, label: str) -> dict:
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
