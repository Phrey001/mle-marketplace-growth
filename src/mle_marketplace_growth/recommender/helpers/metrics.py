from __future__ import annotations

import math

import numpy as np

"""Ranking-metric primitives shared by recommender offline evaluation and serving.

Workflow Steps:
1) Build candidate pools excluding train-seen items.
2) Rank candidate scores by top-k.
3) Compute ranking metrics such as Recall/NDCG/HitRate from ranked item ids.
"""


def _top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """What: Return indices of top-k scores in descending order.
    Why: Shared ranking primitive for model scoring outputs.
    """
    if k >= len(scores):
        return list(np.argsort(-scores))
    partition = np.argpartition(-scores, k)[:k]
    return list(partition[np.argsort(-scores[partition])])


def _user_eval_pool(train_items: set[str], gt_items: set[str], item_to_idx: dict[str, int]) -> tuple[list[int], set[int]] | None:
    """What: Build candidate pool and mapped ground-truth indices for one user.
    Why: Ensures offline evaluation excludes train-seen items and uses valid item ids only.
    """
    ground_truth_indices = {item_to_idx[item_id] for item_id in gt_items if item_id in item_to_idx}
    if not ground_truth_indices:
        return None
    seen = {item_to_idx[item_id] for item_id in train_items if item_id in item_to_idx}
    candidate_item_indices = [idx for idx in range(len(item_to_idx)) if idx not in seen]
    if not candidate_item_indices:
        return None
    return candidate_item_indices, ground_truth_indices


def _ndcg_at_k(ranked_item_indices: list[int], ground_truth_indices: set[int], k: int) -> float:
    """What: Compute NDCG@k for one ranked list and ground-truth set.
    Why: Adds rank-position-sensitive retrieval quality alongside recall/hit-rate.
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


def _evaluate_ranked_items(ranked_item_indices: list[int], ground_truth_indices: set[int], k: int) -> dict[str, float]:
    """What: Compute Recall/NDCG/HitRate for one ranked list and ground-truth set.
    Why: Keeps metric math separate from scoring/ranking orchestration in model selection.
    """
    hits = len(set(ranked_item_indices[:k]).intersection(ground_truth_indices))
    return {
        "recall": hits / len(ground_truth_indices),
        "hit_rate": 1.0 if hits > 0 else 0.0,
        "ndcg": _ndcg_at_k(ranked_item_indices, ground_truth_indices, k),
    }
