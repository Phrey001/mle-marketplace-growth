"""Common helpers for feature-store build orchestration."""

from pathlib import Path

import duckdb


def copy_table_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    output_path: Path,
    count_rows_sql_template: str,
) -> int:
    """Export a DuckDB table to Parquet and return row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY {table_name} TO ? (FORMAT PARQUET)", [str(output_path)])
    count_rows_sql = count_rows_sql_template.replace("{table_name}", table_name)
    row_count = connection.execute(count_rows_sql).fetchone()[0]
    return int(row_count)


def load_shared_silver_table(connection: duckdb.DuckDBPyConnection, silver_path: Path) -> None:
    """Load canonical silver parquet into the current DuckDB session as a temp table.
    Why: Gold builders read the shared silver parquet directly, then keep downstream SQL
    simple by exposing the standard local table name `silver_transactions_line_items`.
    """
    if not silver_path.exists():
        raise FileNotFoundError(f"Shared silver parquet not found: {silver_path}. Run feature_store.build_shared_silver first.")
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE silver_transactions_line_items AS
        SELECT * FROM read_parquet(?);
        """,
        [str(silver_path)],
    )


def run_dq_check(connection: duckdb.DuckDBPyConnection, sql: str, error_message: str) -> None:
    """Run DQ count query and raise when any failing rows are found."""
    failing_rows = int(connection.execute(sql).fetchone()[0] or 0)
    if failing_rows > 0:
        raise ValueError(error_message)


def load_sql_assets(sql_dir: Path) -> dict[str, str]:
    """Load SQL templates required by feature-store build scripts."""
    def load_sql(sql_path: Path) -> str:
        return sql_path.read_text(encoding="utf-8")

    return {
        # Pipeline SQL
        "create_raw_source": load_sql(sql_dir / "pipeline" / "create_raw_source.sql"),
        "raw_bad_timestamp_counts": load_sql(sql_dir / "pipeline" / "raw_bad_timestamp_counts.sql"),
        "silver_duplicate_grain_rows": load_sql(sql_dir / "pipeline" / "silver_duplicate_grain_count.sql"),
        "count_rows": load_sql(sql_dir / "pipeline" / "count_rows.sql"),
        "describe_raw_source": load_sql(sql_dir / "pipeline" / "describe_raw_source.sql"),
        "max_silver_event_date": load_sql(sql_dir / "pipeline" / "max_silver_event_date.sql"),
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
        "dq_invalid_split": load_sql(sql_dir / "dq" / "invalid_split.sql"),
        "dq_split_chronology": load_sql(sql_dir / "dq" / "split_chronology.sql"),
        "dq_user_features_duplicates": load_sql(sql_dir / "dq" / "user_features_duplicates.sql"),
        "dq_invalid_labels": load_sql(sql_dir / "dq" / "invalid_labels.sql"),
        "dq_propensity_train_duplicates": load_sql(sql_dir / "dq" / "propensity_train_duplicates.sql"),
    }
