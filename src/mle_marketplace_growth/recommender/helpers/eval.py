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

Flow Map:
- `_evaluate_model(...)` is the main entry in this module.
- All other functions are helper steps used by `_evaluate_model(...)`.
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


# ===== Main Entry (This Module) =====
# `_evaluate_model(...)` orchestrates all helper functions above.
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
    """What: Evaluate one model family on one split across configured K cutoffs.
    Why: Produces comparable offline retrieval metrics for model selection.

    Pipeline:
    1) Build eligible user candidate pools (`_user_eval_pool`).
    2) Score candidate items for the selected model family.
    3) Accumulate Recall/NDCG/HitRate at each K.
    4) Finalize averaged metrics for this model/split row.
    """
    # ===== Inner Helper Step 1: Build Eligible User Pool =====
    def _eligible_user_pool(
        user_id: str,
    ) -> tuple[list[int], set[int], int] | None:
        """What: Resolve user index plus candidate/ground-truth pools for one user.
        Why: Centralizes eligibility filtering and pool construction in one step.
        """
        if user_id not in split_rows or user_id not in train or user_id not in user_to_idx:
            return None
        pool = _user_eval_pool(train[user_id], split_rows[user_id], item_to_idx)
        if pool is None:
            return None
        candidate_item_indices, ground_truth_indices = pool
        user_index = user_to_idx[user_id]
        return candidate_item_indices, ground_truth_indices, user_index

    # ===== Inner Helper Step 2: Score Candidates =====
    def _score_candidates_for_model(candidate_item_indices: list[int], user_index: int) -> np.ndarray:
        """What: Compute candidate item scores for one user and selected model family.
        Why: Keeps model-family scoring logic local to this evaluation flow.
        """
        # ===== Model Family: Popularity =====
        if model_name == "popularity":
            return np.asarray(popularity[candidate_item_indices])
        # ===== Model Family: Matrix Factorization (MF) =====
        if model_name == "mf":
            return np.asarray(mf_item[candidate_item_indices].dot(mf_user[user_index]))
        # ===== Model Family: Two-Tower =====
        if model_name == "two_tower":
            return np.asarray(tt_item[candidate_item_indices].dot(tt_user[user_index]))
        raise ValueError(f"Unsupported model: {model_name}")

    # ===== Inner Helper Step 3: Accumulate Per-K Metrics =====
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
        Why: Keeps per-K aggregation local to this evaluation flow.
        """
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

        effective_k = min(k, len(candidate_item_indices), item_count)
        top_local_indices = _top_k_indices(candidate_scores, effective_k)
        ranked_item_indices = [candidate_item_indices[idx] for idx in top_local_indices]
        hits = len(set(ranked_item_indices).intersection(ground_truth_indices))
        metric_sums[k]["recall"] += hits / len(ground_truth_indices)
        metric_sums[k]["hit_rate"] += 1.0 if hits > 0 else 0.0
        metric_sums[k]["ndcg"] += _ndcg_at_k(ranked_item_indices, ground_truth_indices, effective_k)

    # ===== Inner Helper Step 4: Finalize Output Metrics =====
    def _finalize_metrics(metric_sums: dict[int, dict[str, float]], top_ks: list[int], eligible_users: int) -> dict:
        """What: Convert aggregated metric sums into averaged metric payload.
        Why: Finalizes one model/split evaluation row.
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

    # ===== Init Aggregation =====
    # This function evaluates exactly one model family (`model_name`) per call.
    metric_sums = {k: {"recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0} for k in top_ks}
    eligible_users = 0
    item_count = len(item_to_idx)

    # ===== User Loop =====
    for user_id in users:
        # Step 1: Build eligible user pool (candidates + ground truth + user index).
        eligible = _eligible_user_pool(user_id)
        if eligible is None:
            continue
        candidate_item_indices, ground_truth_indices, user_index = eligible
        eligible_users += 1

        # Step 2: Score candidate items with the selected model family.
        candidate_scores = _score_candidates_for_model(candidate_item_indices, user_index)
        # candidate_scores shape: [num_candidates]

        # Step 3: Aggregate Recall/NDCG/HitRate at each configured K.
        for k in top_ks:
            _accumulate_ranking_metrics_for_k(
                metric_sums,
                k=k,
                candidate_item_indices=candidate_item_indices,
                ground_truth_indices=ground_truth_indices,
                candidate_scores=candidate_scores,
                item_count=item_count,
            )

    # Step 4: Finalize averaged metrics for this model/split row.
    metrics = _finalize_metrics(metric_sums, top_ks, eligible_users)
    return {"model_name": model_name, "eligible_users": eligible_users, "metrics": metrics}
