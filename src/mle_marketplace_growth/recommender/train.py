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
    EVALUATION_TOP_K,
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


def _load_train_params(cfg: dict) -> RecommenderTrainParams:
    """What: Parse train-time hyperparameters from config into one typed object.
    Why: Centralizes config-field reading so run_train stays focused on pipeline flow;
    semantic and filesystem validation happens separately in `_validate_train_inputs`.
    """
    return RecommenderTrainParams(
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
        mf_components=int(cfg_required(cfg, "mf_components")),
        mf_n_iter=int(cfg_required(cfg, "mf_n_iter")),
        mf_weighting=str(cfg_required(cfg, "mf_weighting")),
    )


def _validate_train_inputs(
    split_path,
    user_index_path,
    item_index_path,
    train_params: RecommenderTrainParams,
) -> None:
    """What: Validate filesystem inputs and semantic training preconditions after config parsing.
    Why: Fails fast with focused error messages before model training starts.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split parquet not found: {split_path}")
    if not user_index_path.exists():
        raise FileNotFoundError(f"User index parquet not found: {user_index_path}")
    if not item_index_path.exists():
        raise FileNotFoundError(f"Item index parquet not found: {item_index_path}")
    if train_params.embedding_dim < 2:
        raise ValueError("embedding_dim must be >= 2")
    if train_params.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if train_params.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if train_params.negative_samples < 0:
        raise ValueError("negative_samples must be >= 0")
    if train_params.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if train_params.l2_reg < 0:
        raise ValueError("l2_reg must be >= 0")
    if train_params.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be >= 0")
    if train_params.early_stop_rounds < 0:
        raise ValueError("early_stop_rounds must be >= 0")
    if train_params.early_stop_k < 1:
        raise ValueError("early_stop_k must be >= 1")
    if train_params.early_stop_tolerance < 0:
        raise ValueError("early_stop_tolerance must be >= 0")
    if train_params.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if train_params.mf_components < 2:
        raise ValueError("mf_components must be >= 2")
    if train_params.mf_n_iter < 1:
        raise ValueError("mf_n_iter must be >= 1")
    if train_params.mf_weighting not in ALLOWED_MF_WEIGHTINGS:
        raise ValueError(f"mf_weighting must be one of {list(ALLOWED_MF_WEIGHTINGS)}")


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
    train_params: RecommenderTrainParams,
) -> CandidateModelOutputs:
    """What: Train popularity, MF, and two-tower candidate models.
    Why: Encapsulates model-fit steps so run_train remains scan-friendly.
    """
    train_pairs = _interaction_pairs(train, user_to_idx, item_to_idx)

    # ===== Model: Popularity =====
    popularity = _popularity_scores(train_pairs, item_count=len(item_to_idx), transform=POPULARITY_TRANSFORM)
    print("[recommender.train] trained popularity baseline")

    # ===== Model: Matrix Factorization (MF) =====
    mf_user, mf_item = _train_mf(
        train_pairs,
        user_count=len(user_to_idx),
        item_count=len(item_to_idx),
        n_components=train_params.mf_components,
        n_iter=train_params.mf_n_iter,
        weighting=train_params.mf_weighting,
        algorithm=MF_ALGORITHM,
        tol=0.0,
    )
    print(
        f"[recommender.train] trained mf baseline (components={train_params.mf_components}, n_iter={train_params.mf_n_iter}, weighting={train_params.mf_weighting}, algorithm={MF_ALGORITHM})"
    )

    # ===== Model: Two-Tower =====
    tt_user, tt_item = _train_two_tower(
        train,
        user_to_idx,
        item_to_idx,
        embedding_dim=train_params.embedding_dim,
        epochs=train_params.epochs,
        learning_rate=train_params.learning_rate,
        negative_samples=train_params.negative_samples,
        batch_size=train_params.batch_size,
        l2_reg=train_params.l2_reg,
        max_grad_norm=train_params.max_grad_norm,
        early_stop_rounds=train_params.early_stop_rounds,
        early_stop_metric=EARLY_STOP_METRIC,
        early_stop_k=train_params.early_stop_k,
        early_stop_tolerance=train_params.early_stop_tolerance,
        validation_interactions=validation,
        temperature=train_params.temperature,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
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
    evaluation_top_k: int,
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
                top_k=evaluation_top_k,
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
    for result in validation_metrics:
        print(
            "[recommender.train] validation:",
            result["model_name"],
            f"Recall@{evaluation_top_k}={result['metrics'].get(f'Recall@{evaluation_top_k}', 0.0):.6f}",
        )
    selected_model_name = max(
        validation_metrics,
        key=lambda result: result["metrics"].get(f"Recall@{evaluation_top_k}", 0.0),
    )["model_name"]
    print(f"[recommender.train] selected_model={selected_model_name} by Recall@{evaluation_top_k}")
    selected_test_row = next((result for result in test_metrics if result["model_name"] == selected_model_name), None)
    if selected_test_row is not None:
        print(
            "[recommender.train] test:",
            selected_model_name,
            f"Recall@{evaluation_top_k}={selected_test_row['metrics'].get(f'Recall@{evaluation_top_k}', 0.0):.6f}",
        )
    return ModelSelectionResult(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        selected_model_name=selected_model_name,
    )


def run_train(config_path: str, output_dir_override=None) -> None:
    """What: Train recommender candidates, select one model, and write artifacts.
    Why: Provides the in-process training entrypoint reused by the CLI and pipeline;
    `output_dir_override` exists for internal experiment utilities such as the tuning sweep.
    """
    # ===== Load Config =====
    runtime = load_recommender_runtime_config(config_path)
    cfg = runtime.cfg
    split_path = runtime.splits_path
    user_index_path = runtime.user_index_path
    item_index_path = runtime.item_index_path
    output_dir = output_dir_override if output_dir_override is not None else runtime.artifacts_dir
    train_params = _load_train_params(cfg)
    # ===== Validate Inputs =====
    _validate_train_inputs(split_path, user_index_path, item_index_path, train_params)

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
        f"embedding_dim={train_params.embedding_dim}, epochs={train_params.epochs}, lr={train_params.learning_rate}, negatives={train_params.negative_samples}, batch_size={train_params.batch_size}, l2={train_params.l2_reg}, max_grad_norm={train_params.max_grad_norm}",
    )
    print(
        "[recommender.train] convergence:",
        f"early_stop_rounds={train_params.early_stop_rounds}, early_stop_metric={EARLY_STOP_METRIC}, early_stop_tolerance={train_params.early_stop_tolerance}, early_stop_k={train_params.early_stop_k}, temperature={train_params.temperature}, normalize_embeddings={NORMALIZE_EMBEDDINGS}, device={DEVICE}, mf_algorithm={MF_ALGORITHM}",
    )

    # ===== Train Models =====
    candidate_outputs = _train_candidate_models(
        train,
        validation,
        user_to_idx,
        item_to_idx,
        train_params,
    )

    # ===== Evaluate Candidates =====
    selection = _evaluate_and_select_model(
        user_ids,
        train,
        validation,
        test,
        user_to_idx,
        item_to_idx,
        EVALUATION_TOP_K,
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
        evaluation_top_k=EVALUATION_TOP_K,
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
            "embedding_dim": train_params.embedding_dim,
            "epochs": train_params.epochs,
            "learning_rate": train_params.learning_rate,
            "negative_samples": train_params.negative_samples,
            "batch_size": train_params.batch_size,
            "l2_reg": train_params.l2_reg,
            "max_grad_norm": train_params.max_grad_norm,
            "early_stop_rounds": train_params.early_stop_rounds,
            "early_stop_metric": EARLY_STOP_METRIC,
            "early_stop_k": train_params.early_stop_k,
            "early_stop_tolerance": train_params.early_stop_tolerance,
            "temperature": train_params.temperature,
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            "device": DEVICE,
            "mf_components": train_params.mf_components,
            "mf_n_iter": train_params.mf_n_iter,
            "mf_weighting": train_params.mf_weighting,
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
