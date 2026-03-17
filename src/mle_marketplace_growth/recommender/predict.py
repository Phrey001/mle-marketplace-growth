"""Generate user-level Top-K recommendations from trained retrieval artifacts.

Workflow Steps:
1) Load runtime config and selected-model artifacts.
2) Select item-side scoring matrix for the chosen model family.
3) Build serving artifacts (item embeddings + ANN index metadata).
4) Score each user and exclude train-seen items.
5) Write ranked Top-K recommendations to CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import faiss
import numpy as np

from mle_marketplace_growth.helpers import read_json, write_json
from mle_marketplace_growth.recommender.constants import ANN_BACKEND
from mle_marketplace_growth.recommender.helpers.artifacts import _write_ann_index
from mle_marketplace_growth.recommender.helpers.config import artifact_paths, load_recommender_runtime_config
from mle_marketplace_growth.recommender.models.mf import MFScorer
from mle_marketplace_growth.recommender.models.popularity import PopularityScorer
from mle_marketplace_growth.recommender.models.two_tower import TwoTowerScorer

SCORER_REGISTRY = {
    "popularity": PopularityScorer,
    "mf": MFScorer,
    "two_tower": TwoTowerScorer,
}


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

def run_predict(config_path: str) -> None:
    """What: Score users to produce Top-K recommender output CSV.
    Why: Reuses frozen selected-model artifacts to generate serving-style retrieval outputs.
    """
    # ===== Load Config =====
    runtime = load_recommender_runtime_config(config_path)
    paths = artifact_paths(runtime)
    selected_model_meta_path = paths.selected_model_meta
    shared_context_path = paths.shared_context
    output_path = paths.topk_recommendations
    top_k = runtime.top_k

    # ===== Validate Inputs =====
    if not selected_model_meta_path.exists():
        raise FileNotFoundError(f"Selected-model metadata not found: {selected_model_meta_path}")
    if not shared_context_path.exists():
        raise FileNotFoundError(f"Shared context not found: {shared_context_path}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    # ===== Load Inputs =====
    selected_model_meta = read_json(selected_model_meta_path)
    shared_context = read_json(shared_context_path)

    selected = selected_model_meta["selected_model_name"]
    user_ids: list[str] = shared_context["user_ids"]
    item_ids: list[str] = shared_context["item_ids"]
    user_to_idx: dict[str, int] = {str(key): int(value) for key, value in shared_context["user_to_idx"].items()}
    train_user_items: dict[str, set[str]] = {
        str(user_id): set(items) for user_id, items in shared_context["train_user_items"].items()
    }
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    model_dir = runtime.artifacts_dir / str(selected_model_meta["model_artifact_dir"])
    scorer_cls = SCORER_REGISTRY.get(selected)
    if scorer_cls is None:
        raise ValueError(f"Unsupported selected model: {selected}")
    scorer = scorer_cls.load_from_dir(model_dir)
    item_matrix = scorer.item_matrix()

    # ===== Build Serving Artifacts =====
    # Serving artifacts are intentionally produced in predict.py (not train_and_select.py)
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
    # Load ANN index only for ANN-backed model families.
    ann_index = (
        _load_ann_index(ann_index_path, ann_meta_path)
        if isinstance(scorer, (MFScorer, TwoTowerScorer))
        else None
    )
    for user_id in user_ids:
        if user_id not in user_to_idx:
            continue
        user_idx = user_to_idx[user_id]
        seen = train_user_items.get(user_id, set())
        seen_indices = {item_to_idx[item_id] for item_id in seen if item_id in item_to_idx}
        ranked_item_indices, ranked_scores = scorer.rank_user_topk(
            user_index=user_idx,
            top_k=top_k,
            item_count=item_count,
            seen_indices=seen_indices,
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
