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

    Inputs:
    - `scores`: one-dimensional score array
    - `k`: number of top positions to return

    Output:
    - list of score-array positions sorted from highest to lowest score
    """
    return list(np.argsort(-scores)[:k])  # negative scores to sort descending because argsort sorts ascending by default


def _build_user_eval_items(
    train_items: set[str],
    held_out_invoice_items: set[str],
    item_to_idx: dict[str, int],
) -> tuple[list[int], set[int]] | None:
    """What: Build candidate item indices and mapped ground-truth indices for one user.
    Why: Ensures offline evaluation excludes train-seen items and uses valid item ids only.

    Inputs:
    - `train_items`: item ids already seen in training for one user
    - `held_out_invoice_items`: item ids from the user's held-out validation/test purchase invoice
    - `item_to_idx`: item-id to scorer-row mapping

    Output:
    - `(candidate_item_indices, ground_truth_indices)` when evaluation is possible
    - `None` when there are no valid held-out targets or no candidate items to rank
    """
    # Map held-out ground-truth item ids into scorer row indices.
    ground_truth_indices = {
        item_to_idx[item_id] for item_id in held_out_invoice_items if item_id in item_to_idx
    }
    if not ground_truth_indices:
        return None

    # Exclude train-seen items from the candidate set for offline evaluation.
    # Use the explicit item-index values here so the local logic does not rely on
    # callers knowing the index space is validated upstream.
    seen_item_indices = {item_to_idx[item_id] for item_id in train_items if item_id in item_to_idx}
    candidate_item_indices = [item_index for item_index in item_to_idx.values() if item_index not in seen_item_indices]
    if not candidate_item_indices:
        return None
    return candidate_item_indices, ground_truth_indices


def _ndcg_at_k(ranked_item_indices: list[int], ground_truth_indices: set[int], k: int) -> float:
    """What: Compute NDCG@k for one ranked list and ground-truth set.
    Why: Adds rank-position-sensitive retrieval quality alongside recall/hit-rate.
    Higher-ranked relevant items contribute more than lower-ranked ones.

    Formula:
    - `DCG@k = sum(1 / log2(rank + 1))` over relevant items appearing in the top-k list
    - `IDCG@k = sum(1 / log2(rank + 1))` for the ideal ordering of `min(|ground_truth|, k)` relevant items
    - `NDCG@k = DCG@k / IDCG@k`

    Inputs:
    - `ranked_item_indices`: ranked predicted item indices for one user
    - `ground_truth_indices`: held-out relevant item indices for the same user
    - `k`: evaluation cutoff

    Output:
    - one NDCG score in the range `[0.0, 1.0]`
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

    Inputs:
    - `ranked_item_indices`: ranked predicted item indices for one user
    - `ground_truth_indices`: held-out relevant item indices for the same user
    - `k`: evaluation cutoff

    Output:
    - metric dict with:
      - `recall`: fraction of held-out relevant items recovered in the top-k list
      - `hit_rate`: whether the top-k list contains at least one held-out relevant item
      - `ndcg`: rank-sensitive quality score that rewards relevant items appearing earlier
    """
    hits = len(set(ranked_item_indices[:k]).intersection(ground_truth_indices))
    return {
        "recall": hits / len(ground_truth_indices),
        "hit_rate": 1.0 if hits > 0 else 0.0,
        "ndcg": _ndcg_at_k(ranked_item_indices, ground_truth_indices, k),
    }
