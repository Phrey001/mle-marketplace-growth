"""Train recommender candidate models on one shared train/validation split.

Workflow Steps:
1) Accept shared split interactions, entity indices, and model-specific params.
2) Train popularity, MF, and two-tower candidates through explicit model-specific paths.
3) Return one in-memory artifact payload per model family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mle_marketplace_growth.recommender.helpers.data import EntityIndex, SplitInteractions
from mle_marketplace_growth.recommender.models.mf import MFTrainParams, train_mf_candidate
from mle_marketplace_growth.recommender.models.popularity import build_train_item_interaction_counts, train_popularity_candidate
from mle_marketplace_growth.recommender.models.two_tower import TwoTowerTrainParams, train_two_tower_candidate


def train_popularity_artifacts(
    *,
    user_item_splits_df: pd.DataFrame,
    item_index_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """What: Train the popularity candidate artifact payload.
    Why: Keeps the popularity training path explicit and separate from MF/two-tower.
    """
    popularity_item_counts = build_train_item_interaction_counts(user_item_splits_df, item_index_df)
    return {"scores": train_popularity_candidate(popularity_item_counts)}


def train_mf_artifacts(
    *,
    split_interactions: SplitInteractions,
    user_index: EntityIndex,
    item_index: EntityIndex,
    mf_params: MFTrainParams,
) -> dict[str, np.ndarray]:
    """What: Train the MF candidate artifact payload.
    Why: Keeps the MF training path explicit and separate from popularity/two-tower.
    """
    mf_user_embeddings, mf_item_embeddings = train_mf_candidate(
        split_interactions.train,
        user_index.id_to_idx,
        item_index.id_to_idx,
        params=mf_params,
    )
    return {
        "user_embeddings": mf_user_embeddings,
        "item_embeddings": mf_item_embeddings,
    }


def train_two_tower_artifacts(
    *,
    split_interactions: SplitInteractions,
    user_index: EntityIndex,
    item_index: EntityIndex,
    two_tower_params: TwoTowerTrainParams,
) -> dict[str, np.ndarray]:
    """What: Train the two-tower candidate artifact payload.
    Why: Keeps the two-tower training path explicit and separate from popularity/MF.
    """
    tt_user_embeddings, tt_item_embeddings = train_two_tower_candidate(
        split_interactions.train,
        split_interactions.validation,
        user_index.id_to_idx,
        item_index.id_to_idx,
        params=two_tower_params,
    )
    return {
        "user_embeddings": tt_user_embeddings,
        "item_embeddings": tt_item_embeddings,
    }
