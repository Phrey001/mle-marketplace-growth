"""Generate user-level Top-K recommendations from trained retrieval model."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import faiss
import numpy as np


def _top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    if k >= len(scores): return list(np.argsort(-scores))
    partition = np.argpartition(-scores, k)[:k]
    return list(partition[np.argsort(-scores[partition])])


def _ann_retrieve_indices(
    ann_index_path: Path,
    ann_meta_path: Path,
    user_vector: np.ndarray,
    args_top_k: int,
    seen_indices: set[int],
    item_count: int,
) -> list[int]:
    if not ann_index_path.exists() or not ann_meta_path.exists(): raise FileNotFoundError("ANN artifacts are required: missing ann_index.bin or ann_index_meta.json.")
    metadata = json.loads(ann_meta_path.read_text(encoding="utf-8"))
    if metadata.get("backend") != "faiss_hnsw_ip": raise ValueError(f"Unsupported ANN backend: {metadata.get('backend')}")

    index = faiss.read_index(str(ann_index_path))
    oversample = min(item_count, max(args_top_k + len(seen_indices) + 20, args_top_k * 3))
    _, indices = index.search(user_vector.astype(np.float32), oversample)
    ranked = []
    for item_idx in indices[0].tolist():
        if item_idx < 0:
            continue
        if item_idx in seen_indices:
            continue
        ranked.append(int(item_idx))
        if len(ranked) >= args_top_k:
            break
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate recommender Top-K predictions.")
    parser.add_argument("--model-bundle", default="artifacts/recommender/model_bundle.pkl", help="Path to model bundle from train.py")
    parser.add_argument("--output-csv", default="artifacts/recommender/topk_recommendations.csv", help="Output Top-K recommendations CSV")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K candidates per user")
    args = parser.parse_args()

    model_path = Path(args.model_bundle)
    if not model_path.exists(): raise FileNotFoundError(f"Model bundle not found: {model_path}")
    if args.top_k < 1: raise ValueError("--top-k must be >= 1")

    with model_path.open("rb") as file:
        bundle = pickle.load(file)

    selected = bundle["selected_model_name"]
    user_ids: list[str] = bundle["user_ids"]
    item_ids: list[str] = bundle["item_ids"]
    user_to_idx: dict[str, int] = bundle["user_to_idx"]
    train_user_items: dict[str, set[str]] = bundle["train_user_items"]
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    popularity = np.asarray(bundle["popularity_scores"])
    mf_user = np.asarray(bundle["mf_user_embeddings"])
    mf_item = np.asarray(bundle["mf_item_embeddings"])
    tt_user = np.asarray(bundle["two_tower_user_embeddings"])
    tt_item = np.asarray(bundle["two_tower_item_embeddings"])
    if selected == "popularity": item_matrix = popularity.reshape(-1, 1)
    elif selected == "mf": item_matrix = mf_item
    elif selected == "two_tower": item_matrix = tt_item
    else: raise ValueError(f"Unsupported selected model: {selected}")

    output_rows: list[list[str | int | float]] = []
    item_count = len(item_ids)
    ann_index_path = model_path.parent / "ann_index.bin"
    ann_meta_path = model_path.parent / "ann_index_meta.json"
    for user_id in user_ids:
        if user_id not in user_to_idx: continue
        user_idx = user_to_idx[user_id]
        seen = train_user_items.get(user_id, set())
        seen_indices = {item_to_idx[item_id] for item_id in seen if item_id in item_to_idx}

        if selected == "popularity":
            candidate_indices = [idx for idx in range(item_count) if idx not in seen_indices]
            if not candidate_indices: continue
            scores = popularity[candidate_indices]
            top_local = _top_k_indices(np.asarray(scores), min(args.top_k, len(candidate_indices)))
            ranked_item_indices = [candidate_indices[idx] for idx in top_local]
            ranked_scores = [float(scores[idx]) for idx in top_local]
        else:
            if len(seen_indices) >= item_count: continue
            if selected == "mf": user_vector = mf_user[user_idx].reshape(1, -1)
            elif selected == "two_tower": user_vector = tt_user[user_idx].reshape(1, -1)
            else: raise ValueError(f"ANN retrieval is not supported for model: {selected}")
            ranked_item_indices = _ann_retrieve_indices(
                ann_index_path=ann_index_path,
                ann_meta_path=ann_meta_path,
                user_vector=user_vector,
                args_top_k=args.top_k,
                seen_indices=seen_indices,
                item_count=item_count,
            )
            ranked_scores = [float(item_matrix[item_idx].dot(user_vector[0])) for item_idx in ranked_item_indices]

        for rank, (item_idx, item_score) in enumerate(zip(ranked_item_indices, ranked_scores, strict=True), start=1):
            output_rows.append([user_id, rank, item_ids[item_idx], round(item_score, 6), selected])

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "rank", "item_id", "score", "model_name"])
        writer.writerows(output_rows)

    print(f"Wrote recommender top-k predictions: {output_path}")


if __name__ == "__main__":
    main()
