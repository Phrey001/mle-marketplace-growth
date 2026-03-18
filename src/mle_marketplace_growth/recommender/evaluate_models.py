"""Offline model evaluation for recommender model comparison.

Workflow Steps:
1) Evaluate each trained model on shared validation and test user pools.
2) Aggregate Recall/NDCG/HitRate metrics at the fixed evaluation cutoff.
3) Return split-specific metric rows for downstream winner selection.
"""

from __future__ import annotations

import numpy as np
from typing import TypedDict

from mle_marketplace_growth.recommender.constants import EVALUATION_TOP_K
from mle_marketplace_growth.recommender.helpers.metrics import _build_user_eval_items, _evaluate_ranked_items, _top_k_indices
from mle_marketplace_growth.recommender.models.mf import MFScorer
from mle_marketplace_growth.recommender.models.popularity import PopularityScorer
from mle_marketplace_growth.recommender.models.two_tower import TwoTowerScorer


class CandidateMetrics(TypedDict):
    model_name: str
    eligible_users: int
    metrics: dict[str, float]


def _evaluate_user(
    *,
    scorer,
    user_id: str,
    train_user_items: dict[str, set[str]],
    held_out_invoice_items_by_user: dict[str, set[str]],
    user_id_to_idx: dict[str, int],
    item_id_to_idx: dict[str, int],
    item_count: int,
) -> dict[str, float] | None:
    """What: Evaluate one user's held-out ranking result for one model scorer.
    Why: Keeps the main model-evaluation loop focused on aggregation rather than per-user steps.

    Held-out truth meaning:
    - `held_out_invoice_items_by_user[user_id]` is the item set from exactly one held-out
      purchase invoice for that user: validation uses the second-latest invoice, test uses the latest.
    """
    if user_id not in held_out_invoice_items_by_user or user_id not in train_user_items or user_id not in user_id_to_idx:
        return None

    user_eval_items = _build_user_eval_items(
        train_user_items[user_id],
        held_out_invoice_items_by_user[user_id],
        item_id_to_idx,
    )
    if user_eval_items is None:
        return None

    candidate_item_indices, ground_truth_indices = user_eval_items
    user_index = user_id_to_idx[user_id]
    candidate_item_scores = scorer.score_candidate_indices(user_index, candidate_item_indices)
    # Use up to the configured cutoff, capped by the available candidate pool.
    effective_k = min(EVALUATION_TOP_K, len(candidate_item_indices), item_count)
    top_candidate_positions = _top_k_indices(candidate_item_scores, effective_k)
    top_candidate_item_indices = [candidate_item_indices[candidate_item_position] for candidate_item_position in top_candidate_positions]
    return _evaluate_ranked_items(top_candidate_item_indices, ground_truth_indices, effective_k)


def _evaluate_model(
    *,
    scorer,
    model_name: str,
    user_ids: list[str],
    train_user_items: dict[str, set[str]],
    held_out_invoice_items_by_user: dict[str, set[str]],
    user_id_to_idx: dict[str, int],
    item_id_to_idx: dict[str, int],
) -> CandidateMetrics:
    """What: Evaluate one model family on one held-out target split.
    Why: Offline ranking-metric logic is shared across models even when training paths differ.

    Inputs:
    - `scorer`: model-specific runtime scorer implementing `score_candidate_indices(...)`
    - `user_ids`: user ids to iterate over for held-out evaluation
    - `train_user_items`: user->seen-item map used to exclude already-observed items
    - `held_out_invoice_items_by_user`: user->held-out-item map where each value is the item set
      from one held-out purchase invoice for validation or test
    - `user_id_to_idx` / `item_id_to_idx`: id-to-row lookups needed by the scorer and eval pool

    Output:
    - one `CandidateMetrics` payload containing the model name, eligible user count,
      and aggregate Recall/NDCG/HitRate at the configured evaluation cutoff

    Candidate-pool policy:
    - Offline evaluation excludes train-seen items for each user.
    - This keeps offline evaluation aligned with the serving-time discovery objective.
    """
    metric_sums = {"recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0}
    eligible_users = 0
    item_count = len(item_id_to_idx)

    for user_id in user_ids:
        user_metrics = _evaluate_user(
            scorer=scorer,
            user_id=user_id,
            train_user_items=train_user_items,
            held_out_invoice_items_by_user=held_out_invoice_items_by_user,
            user_id_to_idx=user_id_to_idx,
            item_id_to_idx=item_id_to_idx,
            item_count=item_count,
        )
        if user_metrics is None:
            continue
        eligible_users += 1
        metric_sums["recall"] += user_metrics["recall"]
        metric_sums["hit_rate"] += user_metrics["hit_rate"]
        metric_sums["ndcg"] += user_metrics["ndcg"]

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


def evaluate_popularity_model(
    *,
    user_ids: list[str],
    train_user_items: dict[str, set[str]],
    target_user_items: dict[str, set[str]],
    user_id_to_idx: dict[str, int],
    item_id_to_idx: dict[str, int],
    popularity_state: dict[str, np.ndarray],
) -> CandidateMetrics:
    scorer = PopularityScorer(scores=popularity_state["scores"])
    return _evaluate_model(
        scorer=scorer,
        model_name="popularity",
        user_ids=user_ids,
        train_user_items=train_user_items,
        held_out_invoice_items_by_user=target_user_items,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
    )


def evaluate_mf_model(
    *,
    user_ids: list[str],
    train_user_items: dict[str, set[str]],
    target_user_items: dict[str, set[str]],
    user_id_to_idx: dict[str, int],
    item_id_to_idx: dict[str, int],
    mf_state: dict[str, np.ndarray],
) -> CandidateMetrics:
    scorer = MFScorer(
        user_embeddings=mf_state["user_embeddings"],
        item_embeddings=mf_state["item_embeddings"],
    )
    return _evaluate_model(
        scorer=scorer,
        model_name="mf",
        user_ids=user_ids,
        train_user_items=train_user_items,
        held_out_invoice_items_by_user=target_user_items,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
    )


def evaluate_two_tower_model(
    *,
    user_ids: list[str],
    train_user_items: dict[str, set[str]],
    target_user_items: dict[str, set[str]],
    user_id_to_idx: dict[str, int],
    item_id_to_idx: dict[str, int],
    two_tower_state: dict[str, np.ndarray],
) -> CandidateMetrics:
    scorer = TwoTowerScorer(
        user_embeddings=two_tower_state["user_embeddings"],
        item_embeddings=two_tower_state["item_embeddings"],
    )
    return _evaluate_model(
        scorer=scorer,
        model_name="two_tower",
        user_ids=user_ids,
        train_user_items=train_user_items,
        held_out_invoice_items_by_user=target_user_items,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
    )
