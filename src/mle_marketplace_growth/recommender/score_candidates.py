"""Offline candidate scoring for recommender model comparison.

Workflow Steps:
1) Score each trained candidate on shared validation and test user pools.
2) Aggregate Recall/NDCG/HitRate metrics at the fixed evaluation cutoff.
3) Return split-specific metric rows for downstream winner selection.
"""

from __future__ import annotations

import numpy as np
from typing import TypedDict

from mle_marketplace_growth.recommender.constants import EVALUATION_TOP_K
from mle_marketplace_growth.recommender.helpers.metrics import _evaluate_ranked_items, _top_k_indices, _user_eval_pool
from mle_marketplace_growth.recommender.models.mf import MFScorer
from mle_marketplace_growth.recommender.models.popularity import PopularityScorer
from mle_marketplace_growth.recommender.models.two_tower import TwoTowerScorer


class CandidateMetrics(TypedDict):
    model_name: str
    eligible_users: int
    metrics: dict[str, float]


def _score_candidate(
    *,
    scorer,
    model_name: str,
    users: list[str],
    train_rows: dict[str, set[str]],
    target_rows: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> CandidateMetrics:
    """What: Score one candidate model family on one held-out target split.
    Why: Offline evaluation logic is shared across models even when training paths differ.
    """
    metric_sums = {"recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0}
    eligible_users = 0
    item_count = len(item_to_idx)

    for user_id in users:
        if user_id not in target_rows or user_id not in train_rows or user_id not in user_to_idx:
            continue
        pool = _user_eval_pool(train_rows[user_id], target_rows[user_id], item_to_idx)
        if pool is None:
            continue
        candidate_item_indices, ground_truth_indices = pool
        user_index = user_to_idx[user_id]
        eligible_users += 1
        candidate_scores = scorer.score_candidate_indices(user_index, candidate_item_indices)
        effective_k = min(EVALUATION_TOP_K, len(candidate_item_indices), item_count)
        top_local_indices = _top_k_indices(candidate_scores, effective_k)
        ranked_item_indices = [candidate_item_indices[idx] for idx in top_local_indices]
        row_metrics = _evaluate_ranked_items(ranked_item_indices, ground_truth_indices, effective_k)
        metric_sums["recall"] += row_metrics["recall"]
        metric_sums["hit_rate"] += row_metrics["hit_rate"]
        metric_sums["ndcg"] += row_metrics["ndcg"]

    if eligible_users == 0:
        metrics = {
            f"Recall@{EVALUATION_TOP_K}": 0.0,
            f"NDCG@{EVALUATION_TOP_K}": 0.0,
            f"HitRate@{EVALUATION_TOP_K}": 0.0,
        }
    else:
        metrics = {
            f"Recall@{EVALUATION_TOP_K}": round(metric_sums["recall"] / eligible_users, 6),
            f"NDCG@{EVALUATION_TOP_K}": round(metric_sums["ndcg"] / eligible_users, 6),
            f"HitRate@{EVALUATION_TOP_K}": round(metric_sums["hit_rate"] / eligible_users, 6),
        }
    return {"model_name": model_name, "eligible_users": eligible_users, "metrics": metrics}


def score_popularity_candidate(
    *,
    users: list[str],
    train_rows: dict[str, set[str]],
    target_rows: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    popularity_artifacts: dict[str, np.ndarray],
) -> CandidateMetrics:
    scorer = PopularityScorer(scores=popularity_artifacts["scores"])
    return _score_candidate(
        scorer=scorer,
        model_name="popularity",
        users=users,
        train_rows=train_rows,
        target_rows=target_rows,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
    )


def score_mf_candidate(
    *,
    users: list[str],
    train_rows: dict[str, set[str]],
    target_rows: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    mf_artifacts: dict[str, np.ndarray],
) -> CandidateMetrics:
    scorer = MFScorer(
        user_embeddings=mf_artifacts["user_embeddings"],
        item_embeddings=mf_artifacts["item_embeddings"],
    )
    return _score_candidate(
        scorer=scorer,
        model_name="mf",
        users=users,
        train_rows=train_rows,
        target_rows=target_rows,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
    )


def score_two_tower_candidate(
    *,
    users: list[str],
    train_rows: dict[str, set[str]],
    target_rows: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    two_tower_artifacts: dict[str, np.ndarray],
) -> CandidateMetrics:
    scorer = TwoTowerScorer(
        user_embeddings=two_tower_artifacts["user_embeddings"],
        item_embeddings=two_tower_artifacts["item_embeddings"],
    )
    return _score_candidate(
        scorer=scorer,
        model_name="two_tower",
        users=users,
        train_rows=train_rows,
        target_rows=target_rows,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
    )
