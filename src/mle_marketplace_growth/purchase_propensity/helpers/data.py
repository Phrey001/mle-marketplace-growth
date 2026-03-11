from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import duckdb


def _quantile(values: np.ndarray | list[float], q: float) -> float:
    """Compute a linear-interpolated quantile for numeric values.
    Used by: train.py, window_sensitivity.py.
    """
    values_arr = np.asarray(values, dtype=float)
    if values_arr.size == 0: raise ValueError("Cannot compute quantile on empty values.")
    return float(np.quantile(values_arr, q, method="linear"))


def _split_df_rows_10_1_1(df: pd.DataFrame, date_column: str = "as_of_date") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Split a DataFrame into strict 10/1/1 train/validation/test partitions."""
    unique_dates = sorted(set(df[date_column].tolist()))
    if len(unique_dates) != 12: raise ValueError("Strict split requires exactly 12 unique as_of_date snapshots " f"(got {len(unique_dates)}).")
    train_dates, validation_dates, test_dates = set(unique_dates[:10]), {unique_dates[10]}, {unique_dates[11]}
    split_desc = (
        f"out_of_time_10_1_1_train_dates={sorted(train_dates)};"
        f"validation_dates={sorted(validation_dates)};"
        f"test_dates={sorted(test_dates)}"
    )
    train_df = df[df[date_column].isin(train_dates)].copy()
    validation_df = df[df[date_column].isin(validation_dates)].copy()
    test_df = df[df[date_column].isin(test_dates)].copy()
    return train_df, validation_df, test_df, split_desc


def _read_parquet_panel(paths: Path | list[Path], allow_empty: bool = False) -> pd.DataFrame:
    """Read one or many parquet files into a single DataFrame."""
    panel_paths = [paths] if isinstance(paths, Path) else paths
    if not panel_paths: raise ValueError("At least one parquet path is required.")
    dfs: list[pd.DataFrame] = []
    connection = duckdb.connect(database=":memory:")
    try:
        for panel_path in panel_paths:
            source_df = connection.execute("SELECT * FROM read_parquet(?)", [str(panel_path)]).fetchdf()
            if not source_df.empty:
                dfs.append(source_df)
    finally:
        connection.close()
    if not dfs and allow_empty:
        return pd.DataFrame()
    if not dfs:
        joined_paths = ", ".join(str(path) for path in panel_paths)
        raise ValueError(f"No rows found in parquet dataset(s): {joined_paths}")
    return pd.concat(dfs, ignore_index=True)


def _load_snapshot_rows(
    input_paths: Path | list[Path],
    feature_columns: list[str],
    purchase_label_column: str,
    revenue_label_column: str,
) -> pd.DataFrame:
    """Load snapshot parquet rows as a typed DataFrame (pre-split).

    purchase_label_column and revenue_label_column select the horizon-specific label columns
    (for example, 30d vs 60d vs 90d) to keep loading aligned with the run config.
    Used by: train.py.
    """
    source_df = _read_parquet_panel(input_paths)
    required_columns = ["user_id", "as_of_date", "country", purchase_label_column, revenue_label_column, *feature_columns]
    missing_columns = [column for column in required_columns if column not in source_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in input dataset(s): {missing_columns}")

    typed_df = source_df[["user_id", "as_of_date", "country", *feature_columns, purchase_label_column, revenue_label_column]].copy()
    typed_df["user_id"] = typed_df["user_id"].astype(str)
    typed_df["as_of_date"] = typed_df["as_of_date"].astype(str)
    typed_df["country"] = typed_df["country"].astype(str)
    typed_df[feature_columns] = typed_df[feature_columns].astype(float)
    typed_df["purchase_label"] = typed_df[purchase_label_column].astype(float)
    typed_df["revenue_label"] = typed_df[revenue_label_column].astype(float)
    return typed_df.drop(columns=[purchase_label_column, revenue_label_column])
