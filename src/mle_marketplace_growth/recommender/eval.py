from __future__ import annotations

import math

import numpy as np


def _top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    if k >= len(scores): return list(np.argsort(-scores))
    partition = np.argpartition(-scores, k)[:k]
    return list(partition[np.argsort(-scores[partition])])


def _user_eval_pool(train_items: set[str], gt_items: set[str], item_to_idx: dict[str, int]) -> tuple[list[int], set[int]] | None:
    gt = {item_to_idx[item_id] for item_id in gt_items if item_id in item_to_idx}
    if not gt: return None
    seen = {item_to_idx[item_id] for item_id in train_items if item_id in item_to_idx}
    candidates = [idx for idx in range(len(item_to_idx)) if idx not in seen]
    if not candidates: return None
    return candidates, gt


def _ndcg_at_k(ranked: list[int], gt: set[int], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(ranked[:k], start=1):
        if item in gt: dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gt), k)
    if ideal_hits == 0: return 0.0
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
    metric_sums = {k: {"recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0} for k in top_ks}
    eligible_users, item_count = 0, len(item_to_idx)

    for user_id in users:
        if user_id not in split_rows or user_id not in train or user_id not in user_to_idx: continue
        pool = _user_eval_pool(train[user_id], split_rows[user_id], item_to_idx)
        if pool is None: continue
        candidate_idx, gt_items = pool
        eligible_users += 1

        user_idx = user_to_idx[user_id]
        if model_name == "popularity": candidate_scores = popularity[candidate_idx]
        elif model_name == "mf": candidate_scores = mf_item[candidate_idx].dot(mf_user[user_idx])
        elif model_name == "two_tower": candidate_scores = tt_item[candidate_idx].dot(tt_user[user_idx])
        else: raise ValueError(f"Unsupported model: {model_name}")

        for k in top_ks:
            k_eff = min(k, len(candidate_idx), item_count)
            top_local = _top_k_indices(np.asarray(candidate_scores), k_eff)
            ranked_items = [candidate_idx[idx] for idx in top_local]
            hits = len(set(ranked_items).intersection(gt_items))
            metric_sums[k]["recall"] += hits / len(gt_items)
            metric_sums[k]["hit_rate"] += 1.0 if hits > 0 else 0.0
            metric_sums[k]["ndcg"] += _ndcg_at_k(ranked_items, gt_items, k_eff)

    metrics = {}
    for k in top_ks:
        if eligible_users == 0:
            metrics[f"Recall@{k}"] = metrics[f"NDCG@{k}"] = metrics[f"HitRate@{k}"] = 0.0
            continue
        metrics[f"Recall@{k}"] = round(metric_sums[k]["recall"] / eligible_users, 6)
        metrics[f"NDCG@{k}"] = round(metric_sums[k]["ndcg"] / eligible_users, 6)
        metrics[f"HitRate@{k}"] = round(metric_sums[k]["hit_rate"] / eligible_users, 6)

    return {"model_name": model_name, "eligible_users": eligible_users, "metrics": metrics}
