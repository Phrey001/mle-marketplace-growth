from __future__ import annotations

from dataclasses import dataclass
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
2) Persist selected-model metadata + model-specific artifacts for prediction reuse.
3) Persist train/validation/test metric JSON payloads.
4) Keep output schemas centralized outside train_and_select/predict scripts.
"""


MODEL_ARTIFACT_FILES = {
    "popularity": ("scores",),
    "mf": ("user_embeddings", "item_embeddings"),
    "two_tower": ("user_embeddings", "item_embeddings"),
}


@dataclass(frozen=True)
class TrainArtifactContext:
    split_path: Path
    evaluation_top_k: int
    user_ids: list[str]
    item_ids: list[str]
    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    train: dict[str, set[str]]
    validation: dict[str, set[str]]
    test: dict[str, set[str]]
    model_config: dict


@dataclass(frozen=True)
class CandidateModelArtifactOutputs:
    artifacts_by_model: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class SelectionArtifactOutputs:
    selected_model_name: str
    validation_metrics: list[dict]
    test_metrics: list[dict]


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
    context: TrainArtifactContext,
    candidate_artifacts: CandidateModelArtifactOutputs,
    selection: SelectionArtifactOutputs,
) -> None:
    """What: Persist selected-model artifacts and training/evaluation metric artifacts.
    Why: Centralizes output schema so the training-stage orchestrator stays focused on model-flow logic.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    models_root = output_dir / "models"
    selected_model_name = selection.selected_model_name
    selected_model_dir = models_root / selected_model_name
    selected_model_dir.mkdir(parents=True, exist_ok=True)

    shared_context = {
        "user_ids": context.user_ids,
        "item_ids": context.item_ids,
        "user_to_idx": context.user_to_idx,
        "item_to_idx": context.item_to_idx,
        "train_user_items": {user_id: sorted(item_ids) for user_id, item_ids in context.train.items()},
        "evaluation_top_k": context.evaluation_top_k,
    }
    write_json(output_dir / "shared_context.json", shared_context)

    selected_model_meta = {
        "selected_model_name": selected_model_name,
        "model_artifact_dir": str(selected_model_dir.relative_to(output_dir)),
        "shared_context_path": "shared_context.json",
        "evaluation_top_k": context.evaluation_top_k,
    }
    artifact_names = MODEL_ARTIFACT_FILES.get(selected_model_name)
    if artifact_names is None:
        raise ValueError(f"Unsupported selected model: {selected_model_name}")
    model_artifacts = candidate_artifacts.artifacts_by_model.get(selected_model_name)
    if model_artifacts is None:
        raise ValueError(f"Missing candidate artifacts for selected model: {selected_model_name}")
    for artifact_name in artifact_names:
        artifact_payload = model_artifacts.get(artifact_name)
        if artifact_payload is None:
            raise ValueError(f"Missing artifact payload '{artifact_name}' for selected model: {selected_model_name}")
        np.save(selected_model_dir / f"{artifact_name}.npy", artifact_payload)
    selected_model_meta["artifact_files"] = [f"{artifact_name}.npy" for artifact_name in artifact_names]
    write_json(output_dir / "selected_model_meta.json", selected_model_meta)

    write_json(
        output_dir / "validation_retrieval_metrics.json",
        {"rows": selection.validation_metrics, "k_value": context.evaluation_top_k},
    )
    write_json(
        output_dir / "test_retrieval_metrics.json",
        {"rows": selection.test_metrics, "k_value": context.evaluation_top_k},
    )
    write_json(
        output_dir / "train_metrics.json",
        {
            "input_splits_path": str(context.split_path),
            "selected_model_name": selected_model_name,
            "selection_rule": f"maximize_validation_Recall@{context.evaluation_top_k}",
            "k_value": context.evaluation_top_k,
            "counts": {
                "users_total": len(context.user_ids),
                "items_train_universe": len(context.item_ids),
                "train_users": len(context.train),
                "validation_users": len(context.validation),
                "test_users": len(context.test),
            },
            "model_config": context.model_config,
        },
    )
