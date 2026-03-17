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
    evaluation_top_k: int
    user_ids: list[str]
    item_ids: list[str]
    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    train: dict[str, set[str]]
    model_config: dict


@dataclass(frozen=True)
class SharedRuntimeContext:
    user_ids: list[str]
    item_ids: list[str]
    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    train_user_items: dict[str, set[str]]
    evaluation_top_k: int


@dataclass(frozen=True)
class SelectedModelMeta:
    selected_model_name: str
    model_artifact_dir: str
    shared_context_path: str
    evaluation_top_k: int
    artifact_files: list[str]


@dataclass(frozen=True)
class RetrievalMetricsPayload:
    rows: list[dict]
    k_value: int


@dataclass(frozen=True)
class CandidateModelArtifactOutputs:
    artifacts_by_model: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class SelectionArtifactOutputs:
    selected_model_name: str
    validation_metrics: list[dict]
    test_metrics: list[dict]


def _shared_runtime_context_payload(context: SharedRuntimeContext) -> dict:
    """What: Convert shared train-time runtime context into a JSON-serializable payload.
    Why: Keeps JSON formatting details out of train/predict stage orchestration.
    """
    return {
        "user_ids": context.user_ids,
        "item_ids": context.item_ids,
        "user_to_idx": context.user_to_idx,
        "item_to_idx": context.item_to_idx,
        "train_user_items": {user_id: sorted(item_ids) for user_id, item_ids in context.train_user_items.items()},
        "evaluation_top_k": context.evaluation_top_k,
    }


def _load_shared_runtime_context(payload: dict) -> SharedRuntimeContext:
    """What: Convert persisted shared runtime JSON into the typed predict-time contract.
    Why: Gives predict.py one explicit object instead of unpacking several loose JSON fields.
    """
    return SharedRuntimeContext(
        user_ids=[str(user_id) for user_id in payload["user_ids"]],
        item_ids=[str(item_id) for item_id in payload["item_ids"]],
        user_to_idx={str(key): int(value) for key, value in payload["user_to_idx"].items()},
        item_to_idx={str(key): int(value) for key, value in payload["item_to_idx"].items()},
        train_user_items={str(user_id): set(items) for user_id, items in payload["train_user_items"].items()},
        evaluation_top_k=int(payload["evaluation_top_k"]),
    )


def _selected_model_meta_payload(meta: SelectedModelMeta) -> dict:
    """What: Convert selected-model metadata into its persisted JSON shape.
    Why: Keeps selected-model artifact schema explicit and centralized.
    """
    return {
        "selected_model_name": meta.selected_model_name,
        "model_artifact_dir": meta.model_artifact_dir,
        "shared_context_path": meta.shared_context_path,
        "evaluation_top_k": meta.evaluation_top_k,
        "artifact_files": meta.artifact_files,
    }


def _load_selected_model_meta(payload: dict) -> SelectedModelMeta:
    """What: Convert selected-model metadata JSON into the typed runtime contract.
    Why: Gives predict/validate stages one explicit model-selection object.
    """
    return SelectedModelMeta(
        selected_model_name=str(payload["selected_model_name"]),
        model_artifact_dir=str(payload["model_artifact_dir"]),
        shared_context_path=str(payload["shared_context_path"]),
        evaluation_top_k=int(payload["evaluation_top_k"]),
        artifact_files=[str(path) for path in payload["artifact_files"]],
    )


def _retrieval_metrics_payload(metrics_payload: RetrievalMetricsPayload) -> dict:
    """What: Convert retrieval metrics into the persisted JSON shape.
    Why: Keeps validation/test metric payload formatting consistent across writers.
    """
    return {"rows": metrics_payload.rows, "k_value": metrics_payload.k_value}


def _load_retrieval_metrics_payload(payload: dict) -> RetrievalMetricsPayload:
    """What: Convert retrieval metrics JSON into the typed validation/report contract.
    Why: Avoids repeated loose-dict access in validation/report code.
    """
    return RetrievalMetricsPayload(rows=list(payload["rows"]), k_value=int(payload["k_value"]))


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


def _write_shared_runtime_context(output_dir: Path, context: TrainArtifactContext) -> None:
    """What: Persist the shared train-time runtime context used later by predict.py.
    Why: Keeps the top-level train-artifact writer visually focused on stage ordering.
    """
    shared_runtime_context = SharedRuntimeContext(
        user_ids=context.user_ids,
        item_ids=context.item_ids,
        user_to_idx=context.user_to_idx,
        item_to_idx=context.item_to_idx,
        train_user_items=context.train,
        evaluation_top_k=context.evaluation_top_k,
    )
    write_json(output_dir / "shared_context.json", _shared_runtime_context_payload(shared_runtime_context))


def _write_selected_model_artifacts(
    output_dir: Path,
    *,
    selected_model_dir: Path,
    context: TrainArtifactContext,
    candidate_artifacts: CandidateModelArtifactOutputs,
    selection: SelectionArtifactOutputs,
) -> None:
    """What: Persist the selected model's artifact payload and selection metadata.
    Why: Keeps selected-model artifact details separate from shared-context and metric writes.
    """
    selected_model_name = selection.selected_model_name
    selected_model_meta = SelectedModelMeta(
        selected_model_name=selected_model_name,
        model_artifact_dir=str(selected_model_dir.relative_to(output_dir)),
        shared_context_path="shared_context.json",
        evaluation_top_k=context.evaluation_top_k,
        artifact_files=[],
    )
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
    write_json(
        output_dir / "selected_model_meta.json",
        _selected_model_meta_payload(
            SelectedModelMeta(
                selected_model_name=selected_model_meta.selected_model_name,
                model_artifact_dir=selected_model_meta.model_artifact_dir,
                shared_context_path=selected_model_meta.shared_context_path,
                evaluation_top_k=selected_model_meta.evaluation_top_k,
                artifact_files=[f"{artifact_name}.npy" for artifact_name in artifact_names],
            )
        ),
    )


def _write_offline_metric_summaries(
    output_dir: Path,
    *,
    context: TrainArtifactContext,
    selection: SelectionArtifactOutputs,
) -> None:
    """What: Persist validation/test retrieval metrics and the train-stage summary JSON.
    Why: Keeps metric/report output details separate from model artifact persistence.
    """
    write_json(
        output_dir / "validation_retrieval_metrics.json",
        _retrieval_metrics_payload(
            RetrievalMetricsPayload(rows=selection.validation_metrics, k_value=context.evaluation_top_k)
        ),
    )
    write_json(
        output_dir / "test_retrieval_metrics.json",
        _retrieval_metrics_payload(RetrievalMetricsPayload(rows=selection.test_metrics, k_value=context.evaluation_top_k)),
    )
    write_json(
        output_dir / "train_metrics.json",
        {
            "selected_model_name": selection.selected_model_name,
            "selection_rule": f"maximize_validation_Recall@{context.evaluation_top_k}",
            "k_value": context.evaluation_top_k,
            "model_config": context.model_config,
        },
    )


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
    # ===== Prepare Output Directories =====
    output_dir.mkdir(parents=True, exist_ok=True)
    models_root = output_dir / "models"
    selected_model_name = selection.selected_model_name
    selected_model_dir = models_root / selected_model_name
    selected_model_dir.mkdir(parents=True, exist_ok=True)

    # ===== Persist Shared Runtime Context =====
    _write_shared_runtime_context(output_dir, context)

    # ===== Persist Selected Model Artifacts =====
    _write_selected_model_artifacts(
        output_dir,
        selected_model_dir=selected_model_dir,
        context=context,
        candidate_artifacts=candidate_artifacts,
        selection=selection,
    )

    # ===== Persist Offline Metric Summaries =====
    _write_offline_metric_summaries(output_dir, context=context, selection=selection)
