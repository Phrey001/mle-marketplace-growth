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

from mle_marketplace_growth.recommender.constants import POPULARITY_TRANSFORM
from mle_marketplace_growth.recommender.helpers.metrics import _top_k_indices


def build_item_interaction_counts(
    train_interactions: dict[str, set[str]],
    item_id_to_idx: dict[str, int],
) -> np.ndarray:
    """What: Count how many train users interacted with each item.
    Why: Popularity training only needs item-level counts, not the full user->items map after this step.
    """
    observed_item_indices = [
        item_id_to_idx[item_id]
        for item_ids in train_interactions.values()
        for item_id in item_ids
        if item_id in item_id_to_idx
    ]
    if not observed_item_indices:
        return np.zeros(len(item_id_to_idx), dtype=float)
    return np.bincount(observed_item_indices, minlength=len(item_id_to_idx)).astype(float)


def _popularity_scores(
    item_interaction_counts: np.ndarray,
    transform: str = "linear",
) -> np.ndarray:
    """What: Compute normalized item popularity scores from item-level interaction counts.
    Why: Popularity scoring should operate on the simplest representation it actually needs.
    """
    scores = np.asarray(item_interaction_counts, dtype=float).copy()
    if transform == "log1p":
        scores = np.log1p(scores)
    elif transform != "linear":
        raise ValueError(f"Unsupported popularity transform: {transform}")
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


@dataclass(frozen=True)
class PopularityScorer:
    """What: Score popularity-model candidates for evaluation and prediction.
    Why: Keeps popularity scoring behavior local to the popularity module.
    """

    scores: np.ndarray
    model_name: str = "popularity"

    def score_candidate_indices(self, user_index: int, candidate_item_indices: list[int]) -> np.ndarray:
        del user_index
        return np.asarray(self.scores[candidate_item_indices])

    def rank_user_topk(
        self,
        *,
        user_index: int,
        top_k: int,
        item_count: int,
        seen_indices: set[int],
        ann_index: faiss.Index | None,
    ) -> tuple[list[int], list[float]]:
        del user_index, ann_index
        candidate_indices = [idx for idx in range(item_count) if idx not in seen_indices]
        if not candidate_indices:
            return [], []
        candidate_scores = self.scores[candidate_indices]
        top_local = _top_k_indices(np.asarray(candidate_scores), min(top_k, len(candidate_indices)))
        ranked_item_indices = [candidate_indices[idx] for idx in top_local]
        ranked_scores = [float(candidate_scores[idx]) for idx in top_local]
        return ranked_item_indices, ranked_scores

    def item_matrix(self) -> np.ndarray:
        return self.scores.reshape(-1, 1)

    @classmethod
    def load_from_dir(cls, model_dir: Path) -> "PopularityScorer":
        """What: Load popularity scorer artifacts from one model directory.
        Why: Keeps popularity artifact-loading logic local to the popularity model module.
        """
        return cls(scores=np.load(model_dir / "scores.npy"))


def train_popularity_candidate(
    item_interaction_counts: np.ndarray,
) -> np.ndarray:
    """What: Train the popularity baseline candidate.
    Why: Keeps popularity-specific fitting in one readable module.
    """
    popularity_scores = _popularity_scores(
        item_interaction_counts,
        transform=POPULARITY_TRANSFORM,
    )
    print("[recommender.models.popularity] trained popularity baseline")
    return popularity_scores
