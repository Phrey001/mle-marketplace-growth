"""Train recommender baselines and a two-tower-style retrieval model.

Workflow Steps:
1) Load config and validate required inputs/hyperparameters.
2) Load split rows and entity-index mappings.
3) Train popularity, MF, and two-tower candidate models.
4) Evaluate all candidates on validation and test splits.
5) Select one model by validation Recall@K.
6) Write model bundle and train/eval artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from mle_marketplace_growth.helpers import cfg_required
from mle_marketplace_growth.recommender.constants import (
    ALLOWED_MF_WEIGHTINGS,
    DEVICE,
    EARLY_STOP_METRIC,
    MF_ALGORITHM,
    MODEL_NAMES,
    NORMALIZE_EMBEDDINGS,
    POPULARITY_TRANSFORM,
)
from mle_marketplace_growth.recommender.helpers.artifacts import _write_train_artifacts
from mle_marketplace_growth.recommender.helpers.config import load_recommender_runtime_config
from mle_marketplace_growth.recommender.helpers.data import _build_interactions, _load_entity_index, _load_split_rows, _validate_split_chronology
from mle_marketplace_growth.recommender.helpers.eval import _evaluate_model
from mle_marketplace_growth.recommender.helpers.models import _interaction_pairs, _popularity_scores, _train_mf, _train_two_tower


@dataclass(frozen=True)
class RecommenderTrainParams:
    embedding_dim: int
    epochs: int
    learning_rate: float
    negative_samples: int
    batch_size: int
    l2_reg: float
    max_grad_norm: float
    early_stop_rounds: int
    early_stop_k: int
    early_stop_tolerance: float
    temperature: float
    tower_hidden_dim: int
    tower_dropout: float
    mf_components: int
    mf_n_iter: int
    mf_weighting: str


@dataclass(frozen=True)
class CandidateModelOutputs:
    popularity: np.ndarray
    mf_user: np.ndarray
    mf_item: np.ndarray
    tt_user: np.ndarray
    tt_item: np.ndarray


@dataclass(frozen=True)
class ModelSelectionResult:
    validation_metrics: list[dict]
    test_metrics: list[dict]
    selected_model_name: str
    select_k: int


def _load_train_params(cfg: dict) -> tuple[RecommenderTrainParams, list[int]]:
    """What: Read train-time hyperparameters from config.
    Why: Centralizes key parsing so run_train stays focused on pipeline flow.
    """
    params = RecommenderTrainParams(
        embedding_dim=int(cfg_required(cfg, "embedding_dim")),
        epochs=int(cfg_required(cfg, "epochs")),
        learning_rate=float(cfg_required(cfg, "learning_rate")),
        negative_samples=int(cfg_required(cfg, "negative_samples")),
        batch_size=int(cfg_required(cfg, "batch_size")),
        l2_reg=float(cfg_required(cfg, "l2_reg")),
        max_grad_norm=float(cfg_required(cfg, "max_grad_norm")),
        early_stop_rounds=int(cfg_required(cfg, "early_stop_rounds")),
        early_stop_k=int(cfg_required(cfg, "early_stop_k")),
        early_stop_tolerance=float(cfg_required(cfg, "early_stop_tolerance")),
        temperature=float(cfg_required(cfg, "temperature")),
        tower_hidden_dim=int(cfg_required(cfg, "tower_hidden_dim")),
        tower_dropout=float(cfg_required(cfg, "tower_dropout")),
        mf_components=int(cfg_required(cfg, "mf_components")),
        mf_n_iter=int(cfg_required(cfg, "mf_n_iter")),
        mf_weighting=str(cfg_required(cfg, "mf_weighting")),
    )
    top_ks_raw = str(cfg_required(cfg, "top_ks"))
    top_ks = sorted({int(value.strip()) for value in top_ks_raw.split(",") if value.strip()})
    return params, top_ks


def _validate_train_inputs(
    split_path,
    user_index_path,
    item_index_path,
    top_ks: list[int],
    train_params: RecommenderTrainParams,
) -> None:
    """What: Validate file inputs and key training parameter constraints.
    Why: Fails fast with focused error messages before model training starts.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split parquet not found: {split_path}")
    if not user_index_path.exists():
        raise FileNotFoundError(f"User index parquet not found: {user_index_path}")
    if not item_index_path.exists():
        raise FileNotFoundError(f"Item index parquet not found: {item_index_path}")
    if not top_ks:
        raise ValueError("At least one K is required in top_ks config")
    if train_params.embedding_dim < 2:
        raise ValueError("embedding_dim must be >= 2")
    if train_params.mf_weighting not in ALLOWED_MF_WEIGHTINGS:
        raise ValueError(f"mf_weighting must be one of {list(ALLOWED_MF_WEIGHTINGS)}")


def _normalized_train_params(train_params: RecommenderTrainParams) -> RecommenderTrainParams:
    """What: Apply deterministic bounds/casts used by training and artifact metadata.
    Why: Keeps normalization logic in one place and avoids duplicated expressions.
    """
    return RecommenderTrainParams(
        embedding_dim=train_params.embedding_dim,
        epochs=train_params.epochs,
        learning_rate=train_params.learning_rate,
        negative_samples=train_params.negative_samples,
        batch_size=max(1, int(train_params.batch_size)),
        l2_reg=train_params.l2_reg,
        max_grad_norm=max(0.0, float(train_params.max_grad_norm)),
        early_stop_rounds=max(0, int(train_params.early_stop_rounds)),
        early_stop_k=max(1, int(train_params.early_stop_k)),
        early_stop_tolerance=max(0.0, float(train_params.early_stop_tolerance)),
        temperature=float(train_params.temperature),
        tower_hidden_dim=max(0, int(train_params.tower_hidden_dim)),
        tower_dropout=max(0.0, float(train_params.tower_dropout)),
        mf_components=train_params.mf_components,
        mf_n_iter=train_params.mf_n_iter,
        mf_weighting=train_params.mf_weighting,
    )


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """What: L2-normalize each row vector.
    Why: Keeps embedding similarity as cosine-style dot product.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _train_candidate_models(
    train: dict[str, set[str]],
    validation: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    normalized_train_params: RecommenderTrainParams,
) -> CandidateModelOutputs:
    """What: Train popularity, MF, and two-tower candidate models.
    Why: Encapsulates model-fit steps so run_train remains scan-friendly.
    """
    params = normalized_train_params
    train_pairs = _interaction_pairs(train, user_to_idx, item_to_idx)

    # ===== Model: Popularity =====
    popularity = _popularity_scores(train_pairs, item_count=len(item_to_idx), transform=POPULARITY_TRANSFORM)
    print("[recommender.train] trained popularity baseline")

    # ===== Model: Matrix Factorization (MF) =====
    mf_user, mf_item = _train_mf(
        train_pairs,
        user_count=len(user_to_idx),
        item_count=len(item_to_idx),
        n_components=params.mf_components,
        n_iter=params.mf_n_iter,
        weighting=params.mf_weighting,
        algorithm=MF_ALGORITHM,
        tol=0.0,
    )
    print(
        f"[recommender.train] trained mf baseline (components={params.mf_components}, n_iter={params.mf_n_iter}, weighting={params.mf_weighting}, algorithm={MF_ALGORITHM})"
    )

    # ===== Model: Two-Tower =====
    tt_user, tt_item = _train_two_tower(
        train,
        user_to_idx,
        item_to_idx,
        embedding_dim=params.embedding_dim,
        epochs=params.epochs,
        learning_rate=params.learning_rate,
        negative_samples=params.negative_samples,
        batch_size=params.batch_size,
        l2_reg=params.l2_reg,
        max_grad_norm=params.max_grad_norm,
        early_stop_rounds=params.early_stop_rounds,
        early_stop_metric=EARLY_STOP_METRIC,
        early_stop_k=params.early_stop_k,
        early_stop_tolerance=params.early_stop_tolerance,
        validation_interactions=validation,
        temperature=params.temperature,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        tower_hidden_dim=params.tower_hidden_dim,
        tower_dropout=params.tower_dropout,
        device=DEVICE,
        verbose=True,
    )
    if NORMALIZE_EMBEDDINGS:
        tt_user, tt_item = _l2_normalize_rows(tt_user), _l2_normalize_rows(tt_item)
    print("[recommender.train] trained two_tower")
    return CandidateModelOutputs(
        popularity=popularity,
        mf_user=mf_user,
        mf_item=mf_item,
        tt_user=tt_user,
        tt_item=tt_item,
    )


def _evaluate_and_select_model(
    user_ids: list[str],
    train: dict[str, set[str]],
    validation: dict[str, set[str]],
    test: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    top_ks: list[int],
    popularity: np.ndarray,
    mf_user: np.ndarray,
    mf_item: np.ndarray,
    tt_user: np.ndarray,
    tt_item: np.ndarray,
) -> ModelSelectionResult:
    """What: Evaluate all candidates and select best by validation Recall@K.
    Why: Freezes one selected model for downstream artifact writing and serving.
    """
    # Evaluate each candidate model in MODEL_NAMES using shared metric logic.
    model_names = list(MODEL_NAMES)
    metrics_by_split = {
        split_name: [
            _evaluate_model(
                model_name=name,
                users=user_ids,
                train=train,
                split_rows=split_rows,
                user_to_idx=user_to_idx,
                item_to_idx=item_to_idx,
                top_ks=top_ks,
                popularity=popularity,
                mf_user=mf_user,
                mf_item=mf_item,
                tt_user=tt_user,
                tt_item=tt_item,
            )
            for name in model_names
        ]
        for split_name, split_rows in [("validation", validation), ("test", test)]
    }
    validation_metrics, test_metrics = metrics_by_split["validation"], metrics_by_split["test"]
    select_k = 20 if 20 in top_ks else max(top_ks)
    for result in validation_metrics:
        print(
            "[recommender.train] validation:",
            result["model_name"],
            f"Recall@{select_k}={result['metrics'].get(f'Recall@{select_k}', 0.0):.6f}",
        )
    selected_model_name = max(
        validation_metrics,
        key=lambda result: result["metrics"].get(f"Recall@{select_k}", 0.0),
    )["model_name"]
    print(f"[recommender.train] selected_model={selected_model_name} by Recall@{select_k}")
    selected_test_row = next((result for result in test_metrics if result["model_name"] == selected_model_name), None)
    if selected_test_row is not None:
        print(
            "[recommender.train] test:",
            selected_model_name,
            f"Recall@{select_k}={selected_test_row['metrics'].get(f'Recall@{select_k}', 0.0):.6f}",
        )
    return ModelSelectionResult(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        selected_model_name=selected_model_name,
        select_k=select_k,
    )


def run_train(config_path: str) -> None:
    """What: Train recommender candidates, select one model, and write artifacts.
    Why: Provides in-process training entrypoint reused by pipeline and CLI.
    """
    # ===== Load Config =====
    runtime = load_recommender_runtime_config(config_path)
    cfg = runtime.cfg
    split_path = runtime.splits_path
    user_index_path = runtime.user_index_path
    item_index_path = runtime.item_index_path
    output_dir = runtime.artifacts_dir
    train_params, top_ks = _load_train_params(cfg)
    # ===== Validate Inputs =====
    _validate_train_inputs(split_path, user_index_path, item_index_path, top_ks, train_params)
    params = _normalized_train_params(train_params)

    # ===== Load Inputs =====
    rows = _load_split_rows(split_path)
    _validate_split_chronology(rows)
    train, validation, test = _build_interactions(rows)
    user_ids, user_to_idx = _load_entity_index(user_index_path, id_col="user_id", idx_col="user_idx")
    item_ids, item_to_idx = _load_entity_index(item_index_path, id_col="item_id", idx_col="item_idx")
    print(
        "[recommender.train] loaded splits:",
        f"train_users={len(train)}, val_users={len(validation)}, test_users={len(test)}, item_universe={len(item_ids)}",
    )
    print(
        "[recommender.train] config:",
        f"embedding_dim={params.embedding_dim}, epochs={params.epochs}, lr={params.learning_rate}, negatives={params.negative_samples}, batch_size={params.batch_size}, l2={params.l2_reg}, max_grad_norm={params.max_grad_norm}",
    )
    print(
        "[recommender.train] convergence:",
        f"early_stop_rounds={params.early_stop_rounds}, early_stop_metric={EARLY_STOP_METRIC}, early_stop_tolerance={params.early_stop_tolerance}, early_stop_k={params.early_stop_k}, temperature={params.temperature}, normalize_embeddings={NORMALIZE_EMBEDDINGS}, tower_hidden_dim={params.tower_hidden_dim}, tower_dropout={params.tower_dropout}, device={DEVICE}, mf_algorithm={MF_ALGORITHM}",
    )

    # ===== Train Models =====
    candidate_outputs = _train_candidate_models(
        train,
        validation,
        user_to_idx,
        item_to_idx,
        params,
    )

    # ===== Evaluate Candidates =====
    selection = _evaluate_and_select_model(
        user_ids,
        train,
        validation,
        test,
        user_to_idx,
        item_to_idx,
        top_ks,
        candidate_outputs.popularity,
        candidate_outputs.mf_user,
        candidate_outputs.mf_item,
        candidate_outputs.tt_user,
        candidate_outputs.tt_item,
    )
    # ===== Write Outputs =====
    _write_train_artifacts(
        output_dir,
        split_path=split_path,
        selected_model_name=selection.selected_model_name,
        select_k=selection.select_k,
        top_ks=top_ks,
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        train=train,
        validation=validation,
        test=test,
        popularity=candidate_outputs.popularity,
        mf_user=candidate_outputs.mf_user,
        mf_item=candidate_outputs.mf_item,
        tt_user=candidate_outputs.tt_user,
        tt_item=candidate_outputs.tt_item,
        validation_metrics=selection.validation_metrics,
        test_metrics=selection.test_metrics,
        model_config={
            "embedding_dim": params.embedding_dim,
            "epochs": params.epochs,
            "learning_rate": params.learning_rate,
            "negative_samples": params.negative_samples,
            "batch_size": params.batch_size,
            "l2_reg": params.l2_reg,
            "max_grad_norm": params.max_grad_norm,
            "early_stop_rounds": params.early_stop_rounds,
            "early_stop_metric": EARLY_STOP_METRIC,
            "early_stop_k": params.early_stop_k,
            "early_stop_tolerance": params.early_stop_tolerance,
            "temperature": params.temperature,
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            "tower_hidden_dim": params.tower_hidden_dim,
            "tower_dropout": params.tower_dropout,
            "device": DEVICE,
            "mf_components": params.mf_components,
            "mf_n_iter": params.mf_n_iter,
            "mf_weighting": params.mf_weighting,
            "mf_algorithm": MF_ALGORITHM,
            "mf_tol": 0.0,
            "popularity_transform": POPULARITY_TRANSFORM,
        },
    )
    print(f"Wrote recommender train artifacts to: {output_dir}")


def main() -> None:
    """What: CLI entrypoint for recommender offline training.
    Why: Enables config-driven runs without exposing many CLI arguments.
    """
    # ===== CLI Args =====
    parser = argparse.ArgumentParser(description="Train recommender retrieval models.")
    parser.add_argument("--config", required=True, help="Recommender YAML config")
    args = parser.parse_args()
    run_train(config_path=args.config)


if __name__ == "__main__":
    main()
