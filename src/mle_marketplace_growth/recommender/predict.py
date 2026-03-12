"""Generate user-level Top-K recommendations from trained retrieval model.

Workflow Steps:
1) Load runtime config and model bundle artifacts.
2) Select item-side scoring matrix for the chosen model family.
3) Build serving artifacts (item embeddings + ANN index metadata).
4) Score each user and exclude train-seen items.
5) Write ranked Top-K recommendations to CSV.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import faiss
import numpy as np

from mle_marketplace_growth.helpers import read_json, write_json
from mle_marketplace_growth.recommender.constants import ANN_BACKEND
from mle_marketplace_growth.recommender.helpers.artifacts import _write_ann_index
from mle_marketplace_growth.recommender.helpers.config import artifact_paths, load_recommender_runtime_config
from mle_marketplace_growth.recommender.helpers.eval import _top_k_indices


def _select_item_matrix(
    selected_model_name: str,
    popularity_scores: np.ndarray,
    mf_item_embeddings: np.ndarray,
    two_tower_item_embeddings: np.ndarray,
) -> np.ndarray:
    """What: Select item-side scoring matrix based on chosen model family.
    Why: Unifies popularity/MF/two-tower serving flow under one retrieval path.
    """
    if selected_model_name == "popularity":
        return popularity_scores.reshape(-1, 1)
    if selected_model_name == "mf":
        return mf_item_embeddings
    if selected_model_name == "two_tower":
        return two_tower_item_embeddings
    raise ValueError(f"Unsupported selected model: {selected_model_name}")


def _prepare_serving_artifacts(
    artifacts_dir: Path,
    selected_model_name: str,
    item_matrix: np.ndarray,
    item_to_idx: dict[str, int],
) -> tuple[Path, Path]:
    """What: Materialize serving artifacts (embeddings/index metadata/ANN metadata).
    Why: Predict stage owns serving outputs, separate from model training outputs.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "item_embeddings.npy", item_matrix)
    write_json(
        artifacts_dir / "item_embedding_index.json",
        {
            "selected_model_name": selected_model_name,
            "item_to_row_index": item_to_idx,
            "embedding_shape": list(item_matrix.shape),
        },
    )
    ann_metadata = _write_ann_index(artifacts_dir, item_matrix)
    ann_meta_path = artifacts_dir / "ann_index_meta.json"
    write_json(ann_meta_path, ann_metadata)
    return artifacts_dir / "ann_index.bin", ann_meta_path


def _load_ann_index(ann_index_path: Path, ann_meta_path: Path) -> faiss.Index:
    """What: Load FAISS index and validate ANN metadata contract.
    Why: Prevents scoring with mismatched or missing ANN artifacts.
    """
    if not ann_index_path.exists() or not ann_meta_path.exists():
        raise FileNotFoundError("ANN artifacts are required: missing ann_index.bin or ann_index_meta.json.")
    metadata = read_json(ann_meta_path)
    if metadata.get("backend") != ANN_BACKEND:
        raise ValueError(f"Unsupported ANN backend: {metadata.get('backend')}")
    return faiss.read_index(str(ann_index_path))


def _ann_retrieve_indices(
    ann_index: faiss.Index,
    user_vector: np.ndarray,
    args_top_k: int,
    seen_indices: set[int],
    item_count: int,
) -> list[int]:
    """What: Retrieve top candidate item indices from ANN and remove seen items.
    Why: Produces serving-style Top-K recommendations for each user.
    """
    oversample = min(item_count, max(args_top_k + len(seen_indices) + 20, args_top_k * 3))
    _, indices = ann_index.search(user_vector.astype(np.float32), oversample)
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


def _score_user_topk(
    *,
    selected_model_name: str,
    user_index: int,
    top_k: int,
    item_count: int,
    seen_indices: set[int],
    popularity_scores: np.ndarray,
    mf_user_embeddings: np.ndarray,
    two_tower_user_embeddings: np.ndarray,
    item_matrix: np.ndarray,
    ann_index: faiss.Index | None,
) -> tuple[list[int], list[float]]:
    """What: Score and rank Top-K item indices for one user.
    Why: Keeps run_predict loop focused on I/O while this helper handles model-specific scoring.
    """
    if selected_model_name == "popularity":
        candidate_indices = [idx for idx in range(item_count) if idx not in seen_indices]
        if not candidate_indices:
            return [], []
        candidate_scores = popularity_scores[candidate_indices]
        top_local = _top_k_indices(np.asarray(candidate_scores), min(top_k, len(candidate_indices)))
        ranked_item_indices = [candidate_indices[idx] for idx in top_local]
        ranked_scores = [float(candidate_scores[idx]) for idx in top_local]
        return ranked_item_indices, ranked_scores

    if len(seen_indices) >= item_count:
        return [], []
    if ann_index is None:
        raise ValueError("ANN index is required for non-popularity models.")
    if selected_model_name == "mf":
        user_vector = mf_user_embeddings[user_index].reshape(1, -1)
    elif selected_model_name == "two_tower":
        user_vector = two_tower_user_embeddings[user_index].reshape(1, -1)
    else:
        raise ValueError(f"ANN retrieval is not supported for model: {selected_model_name}")

    ranked_item_indices = _ann_retrieve_indices(
        ann_index=ann_index,
        user_vector=user_vector,
        args_top_k=top_k,
        seen_indices=seen_indices,
        item_count=item_count,
    )
    ranked_scores = [float(item_matrix[item_idx].dot(user_vector[0])) for item_idx in ranked_item_indices]
    return ranked_item_indices, ranked_scores


def run_predict(config_path: str) -> None:
    """What: Score users to produce Top-K recommender output CSV.
    Why: Reuses trained model bundle to generate serving-style retrieval artifacts.
    """
    # ===== Load Config =====
    runtime = load_recommender_runtime_config(config_path)
    paths = artifact_paths(runtime)
    model_path = paths.model_bundle
    output_path = paths.topk_recommendations
    top_k = runtime.top_k

    # ===== Validate Inputs =====
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    # ===== Load Inputs =====
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
    item_matrix = _select_item_matrix(selected, popularity, mf_item, tt_item)

    # ===== Build Serving Artifacts =====
    # Serving artifacts are intentionally produced in predict.py (not train.py)
    # to keep training and serving responsibilities separate.
    ann_index_path, ann_meta_path = _prepare_serving_artifacts(
        artifacts_dir=runtime.artifacts_dir,
        selected_model_name=selected,
        item_matrix=item_matrix,
        item_to_idx=item_to_idx,
    )

    # ===== Score Users =====
    output_rows: list[list[str | int | float]] = []
    item_count = len(item_ids)
    ann_index = _load_ann_index(ann_index_path, ann_meta_path) if selected in {"mf", "two_tower"} else None
    for user_id in user_ids:
        if user_id not in user_to_idx:
            continue
        user_idx = user_to_idx[user_id]
        seen = train_user_items.get(user_id, set())
        seen_indices = {item_to_idx[item_id] for item_id in seen if item_id in item_to_idx}
        ranked_item_indices, ranked_scores = _score_user_topk(
            selected_model_name=selected,
            user_index=user_idx,
            top_k=top_k,
            item_count=item_count,
            seen_indices=seen_indices,
            popularity_scores=popularity,
            mf_user_embeddings=mf_user,
            two_tower_user_embeddings=tt_user,
            item_matrix=item_matrix,
            ann_index=ann_index,
        )
        if not ranked_item_indices:
            continue

        for rank, (item_idx, item_score) in enumerate(zip(ranked_item_indices, ranked_scores, strict=True), start=1):
            output_rows.append([user_id, rank, item_ids[item_idx], round(item_score, 6), selected])

    # ===== Write Outputs =====
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "rank", "item_id", "score", "model_name"])
        writer.writerows(output_rows)

    print(f"Wrote recommender top-k predictions: {output_path}")


def main() -> None:
    """What: CLI entrypoint for recommender prediction/serving artifact generation.
    Why: Runs scoring with a single config argument for deterministic execution.
    """
    # ===== CLI Args =====
    parser = argparse.ArgumentParser(description="Generate recommender Top-K predictions.")
    parser.add_argument("--config", required=True, help="Recommender YAML config")
    args = parser.parse_args()
    run_predict(config_path=args.config)


if __name__ == "__main__":
    main()
