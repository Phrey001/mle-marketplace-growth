from __future__ import annotations

import math

import numpy as np

"""Evaluation helpers for recommender offline ranking metrics.

Workflow Steps:
1) For each user, build candidate items excluding train-seen items.
2) Score candidates with one model family (popularity / MF / two-tower).
3) Rank top-K items for each configured K.
4) Accumulate Recall@K, NDCG@K, and HitRate@K.
5) Average metrics across eligible users.
"""


def _top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """What: Return indices of top-k scores in descending order.
    Why: Shared ranking primitive for evaluation and serving logic.
    """
    if k >= len(scores):
        return list(np.argsort(-scores))
    partition = np.argpartition(-scores, k)[:k]
    return list(partition[np.argsort(-scores[partition])])


def _user_eval_pool(train_items: set[str], gt_items: set[str], item_to_idx: dict[str, int]) -> tuple[list[int], set[int]] | None:
    """What: Build candidate pool and mapped ground truth indices for one user.
    Why: Ensures evaluation excludes seen train items and uses valid item ids only.
    """
    ground_truth_indices = {item_to_idx[item_id] for item_id in gt_items if item_id in item_to_idx}
    if not ground_truth_indices:
        return None
    seen = {item_to_idx[item_id] for item_id in train_items if item_id in item_to_idx}
    candidate_item_indices = [idx for idx in range(len(item_to_idx)) if idx not in seen]
    if not candidate_item_indices:
        return None
    return candidate_item_indices, ground_truth_indices


def _score_candidates_for_model(
    model_name: str,
    candidate_item_indices: list[int],
    user_index: int,
    popularity: np.ndarray,
    mf_user: np.ndarray,
    mf_item: np.ndarray,
    tt_user: np.ndarray,
    tt_item: np.ndarray,
) -> np.ndarray:
    """What: Compute candidate item scores for one user and one model family.
    Why: Isolates model-specific scoring logic from evaluation-loop bookkeeping.
    """
    if model_name == "popularity":
        return np.asarray(popularity[candidate_item_indices])
    if model_name == "mf":
        return np.asarray(mf_item[candidate_item_indices].dot(mf_user[user_index]))
    if model_name == "two_tower":
        return np.asarray(tt_item[candidate_item_indices].dot(tt_user[user_index]))
    raise ValueError(f"Unsupported model: {model_name}")


def _accumulate_ranking_metrics_for_k(
    metric_sums: dict[int, dict[str, float]],
    *,
    k: int,
    candidate_item_indices: list[int],
    ground_truth_indices: set[int],
    candidate_scores: np.ndarray,
    item_count: int,
) -> None:
    """What: Update Recall/NDCG/HitRate sums for one K value.
    Why: Keeps per-K aggregation logic isolated from user-loop control flow.
    """
    effective_k = min(k, len(candidate_item_indices), item_count)
    top_local_indices = _top_k_indices(candidate_scores, effective_k)
    ranked_item_indices = [candidate_item_indices[idx] for idx in top_local_indices]
    hits = len(set(ranked_item_indices).intersection(ground_truth_indices))
    metric_sums[k]["recall"] += hits / len(ground_truth_indices)
    metric_sums[k]["hit_rate"] += 1.0 if hits > 0 else 0.0
    metric_sums[k]["ndcg"] += _ndcg_at_k(ranked_item_indices, ground_truth_indices, effective_k)


def _finalize_metrics(metric_sums: dict[int, dict[str, float]], top_ks: list[int], eligible_users: int) -> dict:
    """What: Convert aggregated metric sums into averaged metric payload.
    Why: Centralizes final formatting for consistent per-model outputs.
    """
    metrics = {}
    for k in top_ks:
        if eligible_users == 0:
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"NDCG@{k}"] = 0.0
            metrics[f"HitRate@{k}"] = 0.0
            continue
        metrics[f"Recall@{k}"] = round(metric_sums[k]["recall"] / eligible_users, 6)
        metrics[f"NDCG@{k}"] = round(metric_sums[k]["ndcg"] / eligible_users, 6)
        metrics[f"HitRate@{k}"] = round(metric_sums[k]["hit_rate"] / eligible_users, 6)
    return metrics


def _ndcg_at_k(ranked_item_indices: list[int], ground_truth_indices: set[int], k: int) -> float:
    """What: Compute NDCG@k for one ranked list and ground-truth set.
    Why: Adds rank-position-sensitive quality metric alongside recall/hit-rate.
    """
    dcg = 0.0
    for rank_position, item_index in enumerate(ranked_item_indices[:k], start=1):
        if item_index in ground_truth_indices:
            dcg += 1.0 / math.log2(rank_position + 1)
    ideal_hits = min(len(ground_truth_indices), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _evaluate_model(
    model_name: str,
    users: list[str],
    train: dict[str, set[str]],
    split_rows: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    top_ks: list[int],
    popularity: np.ndarray,
    mf_user: np.ndarray,
    mf_item: np.ndarray,
    tt_user: np.ndarray,
    tt_item: np.ndarray,
) -> dict:
    """What: Evaluate one model on one split across configured K cutoffs.
    Why: Produces comparable offline retrieval metrics for model selection.
    """
    # ===== Init Aggregation =====
    metric_sums = {k: {"recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0} for k in top_ks}
    eligible_users = 0
    item_count = len(item_to_idx)

    # ===== User Loop =====
    for user_id in users:
        if user_id not in split_rows or user_id not in train or user_id not in user_to_idx:
            continue
        pool = _user_eval_pool(train[user_id], split_rows[user_id], item_to_idx)
        if pool is None:
            continue
        candidate_item_indices, ground_truth_indices = pool
        eligible_users += 1

        # Score candidate items with the selected model family.
        user_index = user_to_idx[user_id]
        candidate_scores = _score_candidates_for_model(
            model_name=model_name,
            candidate_item_indices=candidate_item_indices,
            user_index=user_index,
            popularity=popularity,
            mf_user=mf_user,
            mf_item=mf_item,
            tt_user=tt_user,
            tt_item=tt_item,
        )
        # candidate_scores shape: [num_candidates]

        # Aggregate Recall/NDCG/HitRate at each configured K.
        for k in top_ks:
            _accumulate_ranking_metrics_for_k(
                metric_sums,
                k=k,
                candidate_item_indices=candidate_item_indices,
                ground_truth_indices=ground_truth_indices,
                candidate_scores=candidate_scores,
                item_count=item_count,
            )

    # ===== Finalize Metrics =====
    metrics = _finalize_metrics(metric_sums, top_ks, eligible_users)
    return {"model_name": model_name, "eligible_users": eligible_users, "metrics": metrics}
