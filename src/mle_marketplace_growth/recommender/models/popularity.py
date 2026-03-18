"""Popularity recommender model module.

Workflow Steps:
1) Read precomputed item interaction counts from train data.
2) Apply the configured popularity-score transform.
3) Normalize the resulting popularity-score vector.
4) Return one item-score vector for shared evaluation/selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from mle_marketplace_growth.recommender.constants import POPULARITY_TRANSFORM
from mle_marketplace_growth.recommender.helpers.metrics import _top_k_indices
from mle_marketplace_growth.recommender.models import RankedItems


@dataclass(frozen=True)
class PopularityScorer:
    """What: Score popularity-model candidates for evaluation and prediction.
    Why: Keeps popularity scoring behavior local to the popularity module.

    This baseline is user-agnostic: every user receives the same item scores
    before seen-item filtering removes already-observed items.

    Method split:
    - `score_candidate_indices(...)` is the offline-evaluation path
    - `rank_user_topk(...)` is the serving/prediction path
    """

    scores: np.ndarray
    model_name: str = "popularity"

    @staticmethod
    def _scores_from_counts(item_interaction_counts: np.ndarray, *, transform: str) -> np.ndarray:
        """What: Compute normalized popularity scores from item-level interaction counts.
        Why: Keeps popularity-specific score construction owned by the popularity model class.
        """
        scores = np.asarray(item_interaction_counts, dtype=float).copy()
        if transform == "log1p":
            scores = np.log1p(scores)  # scores >= 0, since log(1+p) = 0 when p=0
        elif transform != "linear":
            raise ValueError(f"Unsupported popularity transform: {transform}")
        if scores.max() > 0:
            scores = scores / scores.max()  # normalize scores to [0,1], ignores 0 score
        return scores

    # Offline evaluation: score one fixed candidate pool for one user.
    def score_candidate_indices(self, _user_index: int, candidate_item_indices: list[int]) -> np.ndarray:
        """What: Return popularity scores for one user's candidate item indices.
        Why: Keeps the offline-eval scorer contract aligned with MF and two-tower.
        The user index is unused because popularity scores do not depend on the user.
        """
        return np.asarray(self.scores[candidate_item_indices])

    # Serving/prediction: rank one user's unseen items into a top-k list.
    def rank_user_topk(
        self,
        *,
        _user_index: int,
        top_k: int,
        item_count: int,
        seen_indices: set[int],
        _ann_index: faiss.Index | None,
    ) -> RankedItems:
        """What: Rank one user's top-k unseen items by global popularity.
        Why: Uses the same scorer object for prediction after model selection.
        """
        candidate_item_indices = [item_index for item_index in range(item_count) if item_index not in seen_indices]
        if not candidate_item_indices:
            return RankedItems(item_indices=[], scores=[])
        candidate_item_scores = self.scores[candidate_item_indices]
        top_ranked_item_positions = _top_k_indices(np.asarray(candidate_item_scores), min(top_k, len(candidate_item_indices)))
        ranked_item_indices = [candidate_item_indices[item_position] for item_position in top_ranked_item_positions]
        ranked_item_scores = [float(candidate_item_scores[item_position]) for item_position in top_ranked_item_positions]
        return RankedItems(item_indices=ranked_item_indices, scores=ranked_item_scores)

    # Shared runtime utility: used by serving artifact generation across model families.
    def item_matrix(self) -> np.ndarray:
        """What: Expose popularity scores as a single-column item matrix.
        Why: Prediction uses one item-matrix hook across all scorer types.
        """
        return self.scores.reshape(-1, 1)

    # Shared runtime utility: load the scorer from persisted model artifacts.
    @classmethod
    def load_from_dir(cls, model_dir: Path) -> "PopularityScorer":
        """What: Load popularity scorer artifacts from one model directory.
        Why: Keeps popularity artifact-loading logic local to the popularity model module.
        """
        return cls(scores=np.load(model_dir / "scores.npy"))

    @classmethod
    def from_train_splits(
        cls,
        user_item_splits_df: pd.DataFrame,
        item_index_df: pd.DataFrame,
    ) -> "PopularityScorer":
        """What: Build the popularity scorer directly from train split rows and the item index table.
        Why: Keeps popularity-specific training construction owned by the popularity model class.
        """
        train_item_indices = (
            user_item_splits_df.loc[user_item_splits_df["split"] == "train", ["item_id"]]
            .merge(item_index_df[["item_id", "item_idx"]], on="item_id", how="inner")
            ["item_idx"]
            .astype(int)
            .to_numpy()
        )
        if train_item_indices.size == 0:
            item_interaction_counts = np.zeros(len(item_index_df), dtype=float)
        else:
            item_interaction_counts = np.bincount(train_item_indices, minlength=len(item_index_df)).astype(float)
        popularity_scores = cls._scores_from_counts(item_interaction_counts, transform=POPULARITY_TRANSFORM)
        print("[recommender.models.popularity] trained popularity baseline")
        return cls(scores=popularity_scores)
