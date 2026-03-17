from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SplitInteractions:
    train: dict[str, set[str]]
    validation: dict[str, set[str]]
    test: dict[str, set[str]]


@dataclass(frozen=True)
class EntityIndex:
    ids: list[str]
    id_to_idx: dict[str, int]


# ===== Schema Checks =====

def _require_columns(rows_df: pd.DataFrame, required: set[str], *, label: str) -> None:
    """What: Assert a DataFrame contains the required columns for one contract.
    Why: Keeps the loaders focused on flow rather than repeated schema-check boilerplate.
    """
    missing = sorted(required - set(rows_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def _validate_allowed_splits(split_series: pd.Series) -> None:
    """What: Assert split labels use the canonical train/val/test vocabulary.
    Why: Keeps downstream ML code free of split-label normalization or branching.
    """
    observed_splits = set(split_series.astype(str).str.strip().str.lower())
    allowed_splits = {"train", "val", "test"}
    if not observed_splits.issubset(allowed_splits):
        invalid_splits = sorted(observed_splits - allowed_splits)
        raise ValueError(f"Invalid split values in split dataset: {invalid_splits}")


def _validate_unique_contiguous_index(indexed_df: pd.DataFrame, *, id_col: str, idx_col: str, path: Path) -> None:
    """What: Validate entity ids are unique and integer indices run contiguously from zero.
    Why: Guarantees stable embedding-matrix alignment for model training and serving artifacts.
    """
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


# ===== Split-Table Loading =====

def _read_parquet_to_df(path: Path) -> pd.DataFrame:
    """What: Load parquet rows into a pandas DataFrame via DuckDB.
    Why: Keeps downstream ML prep on columnar/tabular structures.
    """
    connection = duckdb.connect(database=":memory:")
    try:
        return connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetch_df()
    finally:
        connection.close()


def _load_user_item_splits_df(path: Path) -> pd.DataFrame:
    """What: Load `user_item_splits.parquet` and validate required split-table columns.
    Why: Ensures train/eval logic receives the expected split schema.
    """
    user_item_splits_df = _read_parquet_to_df(path)
    if user_item_splits_df.empty:
        raise ValueError(f"No rows found in split dataset: {path}")
    _require_columns(user_item_splits_df, {"user_id", "item_id", "split", "event_ts"}, label="split dataset")
    _validate_allowed_splits(user_item_splits_df["split"])
    return user_item_splits_df


# ===== Split-Table Transforms =====

def _validate_split_chronology(user_item_splits_df: pd.DataFrame) -> None:
    """What: Enforce per-user train < validation < test temporal ordering.
    Why: Prevents temporal leakage after the feature store assigns invoice-level
    chronological splits and all line items inherit that split.
    """
    if user_item_splits_df.empty:
        return

    user_item_splits_with_ts_df = user_item_splits_df.copy()
    user_item_splits_with_ts_df["event_ts"] = pd.to_datetime(user_item_splits_with_ts_df["event_ts"], errors="coerce")
    if user_item_splits_with_ts_df["event_ts"].isna().any():
        raise ValueError("Invalid event_ts in split dataset; failed to parse datetime.")

    # Collapse each user's rows into one summary row:
    # user_id | max_event_ts_train | min_event_ts_val | max_event_ts_val | min_event_ts_test
    # The chronology rule is then: train_end <= val_start and val_end <= test_start.
    per_user_split_bounds_df = (
        user_item_splits_with_ts_df.groupby(["user_id", "split"], sort=False)["event_ts"]
        .agg(min_event_ts="min", max_event_ts="max")
        .unstack("split")
    )
    per_user_split_bounds_df.columns = [
        f"{agg_name}_{split_name}"
        for agg_name, split_name in per_user_split_bounds_df.columns.to_flat_index()
    ]

    required_split_bounds = {
        "max_event_ts_train",
        "min_event_ts_val",
        "max_event_ts_val",
        "min_event_ts_test",
    }
    if not required_split_bounds.issubset(per_user_split_bounds_df.columns):
        return

    users_with_train_val_test_df = per_user_split_bounds_df.dropna(subset=sorted(required_split_bounds))
    users_with_invalid_split_order = (
        (users_with_train_val_test_df["max_event_ts_train"] > users_with_train_val_test_df["min_event_ts_val"])
        | (users_with_train_val_test_df["max_event_ts_val"] > users_with_train_val_test_df["min_event_ts_test"])
    )
    if users_with_invalid_split_order.any():
        raise ValueError(
            "Split chronology violation detected: expected train_end <= val_start and val_end <= test_start per user."
        )


def _build_split_interactions(
    user_item_splits_df: pd.DataFrame,
) -> SplitInteractions:
    """What: Convert split rows into user->item interaction maps by split.
    Why: Provides compact structures for model training and metric evaluation
    after invoice-level split decisions have already been assigned upstream.
    """
    user_item_split_labels_df = user_item_splits_df[["user_id", "item_id", "split"]].copy()

    def _interaction_map(split_values: set[str]) -> dict[str, set[str]]:
        split_df = user_item_split_labels_df[user_item_split_labels_df["split"].isin(split_values)]
        grouped = split_df.groupby("user_id", sort=False)["item_id"].agg(set)
        return grouped.to_dict()

    train = _interaction_map({"train"})
    validation = _interaction_map({"val"})
    test = _interaction_map({"test"})
    if not train:
        raise ValueError("No train interactions found.")
    if not validation:
        raise ValueError("No validation interactions found.")
    if not test:
        raise ValueError("No test interactions found.")
    return SplitInteractions(train=train, validation=validation, test=test)


# ===== Entity-Index Loaders =====

def _load_entity_index(path: Path, id_col: str, idx_col: str) -> EntityIndex:
    """What: Load entity index table and validate unique contiguous indices.
    Why: Guarantees stable row-index mapping for embedding matrices.
    """
    entity_index_df = _read_parquet_to_df(path)
    if entity_index_df.empty:
        raise ValueError(f"No rows found in entity index file: {path}")
    _require_columns(entity_index_df, {id_col, idx_col}, label=f"entity index file {path}")
    indexed_df = entity_index_df[[id_col, idx_col]].copy()
    indexed_df[idx_col] = indexed_df[idx_col].astype(int)
    _validate_unique_contiguous_index(indexed_df, id_col=id_col, idx_col=idx_col, path=path)

    # Preserve row-id alignment by explicit index reordering: idx 0..N-1 -> entity id.
    expected = np.arange(len(indexed_df), dtype=int)
    ordered_ids = (
        indexed_df.set_index(idx_col)[id_col]
        .reindex(expected)
        .tolist()
    )
    id_to_idx = indexed_df.set_index(id_col)[idx_col].to_dict()
    return EntityIndex(ids=ordered_ids, id_to_idx=id_to_idx)


def _load_user_index(path: Path) -> EntityIndex:
    """What: Load the user index table as the normalized user-index contract.
    Why: Makes user-index loading explicit at the call site.
    """
    return _load_entity_index(path, id_col="user_id", idx_col="user_idx")


def _load_item_index(path: Path) -> EntityIndex:
    """What: Load the item index table as the normalized item-index contract.
    Why: Makes item-index loading explicit at the call site.
    """
    return _load_entity_index(path, id_col="item_id", idx_col="item_idx")


def _load_entity_index_df(path: Path, id_col: str, idx_col: str) -> pd.DataFrame:
    """What: Load entity index table as a normalized DataFrame with validated contiguous indices.
    Why: Some model paths are more readable with pandas-native joins than with dict lookups.
    """
    entity_index_df = _read_parquet_to_df(path)
    if entity_index_df.empty:
        raise ValueError(f"No rows found in entity index file: {path}")
    _require_columns(entity_index_df, {id_col, idx_col}, label=f"entity index file {path}")
    indexed_df = entity_index_df[[id_col, idx_col]].copy()
    indexed_df[idx_col] = indexed_df[idx_col].astype(int)
    _validate_unique_contiguous_index(indexed_df, id_col=id_col, idx_col=idx_col, path=path)
    return indexed_df


def _load_user_index_df(path: Path) -> pd.DataFrame:
    """What: Load the user index table as a normalized DataFrame.
    Why: Makes user-index dataframe use explicit at the call site.
    """
    return _load_entity_index_df(path, id_col="user_id", idx_col="user_idx")


def _load_item_index_df(path: Path) -> pd.DataFrame:
    """What: Load the item index table as a normalized DataFrame.
    Why: Makes item-index dataframe use explicit at the call site.
    """
    return _load_entity_index_df(path, id_col="item_id", idx_col="item_idx")
