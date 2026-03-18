"""Train recommender candidate models on one shared train/validation split.

Workflow Steps:
1) Accept shared split interactions, entity indices, and model-specific params.
2) Train popularity, MF, and two-tower candidates through explicit model-specific paths.
3) Return one in-memory model-state payload per model family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mle_marketplace_growth.recommender.helpers.data import EntityIndex, SplitInteractions
from mle_marketplace_growth.recommender.models.mf import MFScorer, MFTrainParams
from mle_marketplace_growth.recommender.models.popularity import PopularityScorer
from mle_marketplace_growth.recommender.models.two_tower import TwoTowerTrainParams, train_two_tower_candidate


def train_popularity_state(
    *,
    user_item_splits_df: pd.DataFrame,
    item_index_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """What: Train the popularity model state payload.
    Why: Keeps the popularity training path explicit and separate from MF/two-tower.
    """
    popularity_scorer = PopularityScorer.from_train_splits(user_item_splits_df, item_index_df)
    return {"scores": popularity_scorer.scores}


def train_mf_state(
    *,
    split_interactions: SplitInteractions,
    user_index: EntityIndex,
    item_index: EntityIndex,
    mf_params: MFTrainParams,
) -> dict[str, np.ndarray]:
    """What: Train the MF model state payload.
    Why: Keeps the MF training path explicit and separate from popularity/two-tower.
    """
    mf_scorer = MFScorer.from_train_interactions(
        split_interactions.train,
        user_index.id_to_idx,
        item_index.id_to_idx,
        params=mf_params,
    )
    return {
        "user_embeddings": mf_scorer.user_embeddings,
        "item_embeddings": mf_scorer.item_embeddings,
    }


def train_two_tower_state(
    *,
    split_interactions: SplitInteractions,
    user_index: EntityIndex,
    item_index: EntityIndex,
    two_tower_params: TwoTowerTrainParams,
) -> dict[str, np.ndarray]:
    """What: Train the two-tower model state payload.
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
