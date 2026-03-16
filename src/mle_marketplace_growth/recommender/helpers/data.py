from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

"""Data-loading helpers for recommender train/eval inputs.

Workflow Steps:
1) Read parquet rows through DuckDB into pandas DataFrames.
2) Validate split schema and per-user chronology constraints.
3) Build train/validation/test interaction maps.
4) Load and validate contiguous user/item index tables.
"""


def _read_table(path: Path) -> pd.DataFrame:
    """What: Load parquet rows into a pandas DataFrame via DuckDB.
    Why: Keeps downstream ML prep on columnar/tabular structures.
    """
    connection = duckdb.connect(database=":memory:")
    try:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetch_df()
    finally:
        connection.close()


def _load_split_rows(path: Path) -> pd.DataFrame:
    """What: Load split parquet and validate required split-table columns.
    Why: Ensures train/eval logic receives the expected split schema.
    """
    rows_df = _read_table(path)
    if rows_df.empty:
        raise ValueError(f"No rows found in split dataset: {path}")
    required = {"user_id", "item_id", "split", "event_ts"}
    missing = sorted(required - set(rows_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in split dataset: {missing}")
    return rows_df


def _validate_split_chronology(rows_df: pd.DataFrame) -> None:
    """What: Enforce per-user train < validation < test temporal ordering.
    Why: Prevents temporal leakage after the feature store assigns invoice-level
    chronological splits and all line items inherit that split.
    """
    normalized_df = rows_df.copy()
    normalized_df["split_norm"] = normalized_df["split"].astype(str).str.strip().str.lower().replace({"validation": "val"})
    normalized_df = normalized_df[normalized_df["split_norm"].isin({"train", "val", "test"})]
    if normalized_df.empty:
        return

    normalized_df["event_ts"] = pd.to_datetime(normalized_df["event_ts"], errors="coerce")
    if normalized_df["event_ts"].isna().any():
        raise ValueError("Invalid event_ts in split dataset; failed to parse datetime.")

    violations = 0
    for _, user_df in normalized_df.groupby("user_id", sort=False):
        train_ts = user_df.loc[user_df["split_norm"] == "train", "event_ts"]
        val_ts = user_df.loc[user_df["split_norm"] == "val", "event_ts"]
        test_ts = user_df.loc[user_df["split_norm"] == "test", "event_ts"]
        if train_ts.empty or val_ts.empty or test_ts.empty:
            continue
        if train_ts.max() > val_ts.min():
            violations += 1
        if train_ts.max() > test_ts.min():
            violations += 1
        if val_ts.min() > test_ts.min():
            violations += 1
    if violations > 0:
        raise ValueError(f"Split chronology violation detected: {violations} user-level ordering violations.")


def _build_interactions(rows_df: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """What: Convert split rows into user->item interaction maps by split.
    Why: Provides compact structures for model training and metric evaluation
    after invoice-level split decisions have already been assigned upstream.
    """
    normalized_df = rows_df[["user_id", "item_id", "split"]].copy()
    normalized_df["split_norm"] = normalized_df["split"].astype(str).str.strip().str.lower().replace({"validation": "val"})

    def _interaction_map(split_values: set[str]) -> dict[str, set[str]]:
        split_df = normalized_df[normalized_df["split_norm"].isin(split_values)]
        grouped = split_df.groupby("user_id", sort=False)["item_id"].agg(lambda items: set(items.tolist()))
        return grouped.to_dict()

    train = _interaction_map({"train"})
    validation = _interaction_map({"validation", "val"})
    test = _interaction_map({"test"})
    if not train:
        raise ValueError("No train interactions found.")
    if not validation:
        raise ValueError("No validation interactions found.")
    if not test:
        raise ValueError("No test interactions found.")
    return train, validation, test


def _load_entity_index(path: Path, id_col: str, idx_col: str) -> tuple[list[str], dict[str, int]]:
    """What: Load entity index table and validate unique contiguous indices.
    Why: Guarantees stable row-index mapping for embedding matrices.
    """
    if not path.exists(): raise FileNotFoundError(f"Entity index file not found: {path}")
    rows_df = _read_table(path)
    if rows_df.empty:
        raise ValueError(f"No rows found in entity index file: {path}")
    required = {id_col, idx_col}
    missing = sorted(required - set(rows_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in entity index file {path}: {missing}")
    indexed_df = rows_df[[id_col, idx_col]].copy()
    indexed_df[idx_col] = indexed_df[idx_col].astype(int)

    if indexed_df[id_col].duplicated().any():
        duplicate_id = indexed_df.loc[indexed_df[id_col].duplicated(), id_col].iloc[0]
        raise ValueError(f"Duplicate entity id in index file {path}: {duplicate_id}")
    if indexed_df[idx_col].duplicated().any():
        duplicate_idx = indexed_df.loc[indexed_df[idx_col].duplicated(), idx_col].iloc[0]
        raise ValueError(f"Duplicate entity idx in index file {path}: {duplicate_idx}")

    observed = np.sort(indexed_df[idx_col].to_numpy())
    expected = np.arange(len(indexed_df), dtype=int)
    if not np.array_equal(observed, expected):
        raise ValueError(f"Entity idx must be contiguous from 0 in {path}")

    # Preserve row-id alignment by explicit index reordering: idx 0..N-1 -> entity id.
    ordered_ids = (
        indexed_df.set_index(idx_col)[id_col]
        .reindex(expected)
        .tolist()
    )
    id_to_idx = indexed_df.set_index(id_col)[idx_col].to_dict()
    return ordered_ids, id_to_idx
