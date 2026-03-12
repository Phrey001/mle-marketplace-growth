from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from mle_marketplace_growth.helpers import write_json
from mle_marketplace_growth.recommender.constants import (
    ANN_BACKEND,
    ANN_HNSW_EF_CONSTRUCTION,
    ANN_HNSW_M,
    ANN_METRIC,
)

"""Artifact-writing helpers for recommender training and serving outputs.

Workflow Steps:
1) Persist ANN index artifacts from item embeddings.
2) Persist selected-model bundle for prediction stage reuse.
3) Persist train/validation/test metric JSON payloads.
4) Keep output schemas centralized outside train/predict scripts.
"""


def _write_ann_index(output_dir: Path, item_embeddings: np.ndarray) -> dict:
    """What: Build and persist FAISS HNSW ANN index for item embeddings.
    Why: Enables fast approximate nearest-neighbor retrieval during predict stage.
    """
    ann_index_path = output_dir / "ann_index.bin"
    embeddings = np.asarray(item_embeddings, dtype=np.float32)
    dim = int(embeddings.shape[1])
    index = faiss.IndexHNSWFlat(dim, ANN_HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ANN_HNSW_EF_CONSTRUCTION
    index.add(embeddings)
    faiss.write_index(index, str(ann_index_path))
    return {
        "backend": ANN_BACKEND,
        "metric": ANN_METRIC,
        "dimension": dim,
        "item_count": int(embeddings.shape[0]),
        "notes": "ANN retrieval enabled via FAISS HNSW index.",
    }


def _write_train_artifacts(
    output_dir: Path,
    *,
    split_path: Path,
    selected_model_name: str,
    select_k: int,
    top_ks: list[int],
    user_ids: list[str],
    item_ids: list[str],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    train: dict[str, set[str]],
    validation: dict[str, set[str]],
    test: dict[str, set[str]],
    popularity: np.ndarray,
    mf_user: np.ndarray,
    mf_item: np.ndarray,
    tt_user: np.ndarray,
    tt_item: np.ndarray,
    validation_metrics: list[dict],
    test_metrics: list[dict],
    model_config: dict,
) -> None:
    """What: Persist model bundle and training/evaluation metric artifacts.
    Why: Centralizes output schema so train.py focuses on model-flow logic.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "model_bundle.pkl").open("wb") as file:
        pickle.dump(
            {
                "selected_model_name": selected_model_name,
                "user_ids": user_ids,
                "item_ids": item_ids,
                "user_to_idx": user_to_idx,
                "item_to_idx": item_to_idx,
                "train_user_items": train,
                "top_ks": top_ks,
                "popularity_scores": popularity,
                "mf_user_embeddings": mf_user,
                "mf_item_embeddings": mf_item,
                "two_tower_user_embeddings": tt_user,
                "two_tower_item_embeddings": tt_item,
            },
            file,
        )

    write_json(output_dir / "validation_retrieval_metrics.json", {"rows": validation_metrics, "k_values": top_ks})
    write_json(output_dir / "test_retrieval_metrics.json", {"rows": test_metrics, "k_values": top_ks})
    write_json(
        output_dir / "train_metrics.json",
        {
            "input_splits_path": str(split_path),
            "selected_model_name": selected_model_name,
            "selection_rule": f"maximize_validation_Recall@{select_k}",
            "k_values": top_ks,
            "counts": {
                "users_total": len(user_ids),
                "items_train_universe": len(item_ids),
                "train_users": len(train),
                "validation_users": len(validation),
                "test_users": len(test),
            },
            "model_config": model_config,
        },
    )
