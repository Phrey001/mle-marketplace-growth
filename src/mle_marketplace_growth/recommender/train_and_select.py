"""Train recommender candidates, evaluate them offline, and select one winner.

Workflow Steps:
1) Load config and validate required inputs/hyperparameters.
2) Load split rows and entity-index mappings.
3) Train popularity, MF, and two-tower candidates in separate model modules.
4) Evaluate all candidates on validation and test splits.
5) Select one model by validation Recall@K.
6) Write selected-model artifacts and train/eval metrics.

This module is the owner of offline model evaluation in the recommender pipeline.
`helpers/metrics.py` provides the ranking and metric primitives used by this stage.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from mle_marketplace_growth.helpers import cfg_required
from mle_marketplace_growth.recommender.constants import ALLOWED_MF_WEIGHTINGS, EVALUATION_TOP_K
from mle_marketplace_growth.recommender.contracts import RecommenderTrainParams, TrainingInputs
from mle_marketplace_growth.recommender.helpers.artifacts import (
    CandidateModelArtifactOutputs,
    SelectionArtifactOutputs,
    TrainArtifactContext,
    _write_train_artifacts,
)
from mle_marketplace_growth.recommender.helpers.config import load_recommender_runtime_config
from mle_marketplace_growth.recommender.helpers.data import (
    _build_split_interactions,
    _load_entity_index,
    _load_user_item_splits_df,
    _validate_split_chronology,
)
from mle_marketplace_growth.recommender.select_best_model import evaluate_and_select_model
from mle_marketplace_growth.recommender.train_candidates import train_candidate_models


def _load_train_params(cfg: dict) -> RecommenderTrainParams:
    """What: Parse train-time hyperparameters from config into one typed object.
    Why: Centralizes config-field reading so run_train_and_select stays focused on pipeline flow;
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
    user_item_splits_path,
    user_index_path,
    item_index_path,
    train_params: RecommenderTrainParams,
) -> None:
    """What: Validate filesystem inputs and semantic training preconditions after config parsing.
    Why: Fails fast with focused error messages before candidate training starts.
    """
    if not user_item_splits_path.exists():
        raise FileNotFoundError(f"Split parquet not found: {user_item_splits_path}")
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


def _load_training_inputs(user_item_splits_path, user_index_path, item_index_path) -> TrainingInputs:
    """What: Load and normalize all shared inputs needed by candidate training/evaluation.
    Why: Keeps the top-level stage entrypoint focused on orchestration rather than setup details.
    """
    user_item_splits_df = _load_user_item_splits_df(user_item_splits_path)
    _validate_split_chronology(user_item_splits_df)
    split_interactions = _build_split_interactions(user_item_splits_df)
    user_index = _load_entity_index(user_index_path, id_col="user_id", idx_col="user_idx")
    item_index = _load_entity_index(item_index_path, id_col="item_id", idx_col="item_idx")
    return TrainingInputs(
        split_interactions=split_interactions,
        user_index=user_index,
        item_index=item_index,
    )


def _log_training_context(training_inputs: TrainingInputs, train_params: RecommenderTrainParams) -> None:
    """What: Print concise run context before candidate training starts.
    Why: Makes the stage easier to inspect without scattering print statements across the main flow.
    """
    print(
        "[recommender.train_and_select] loaded splits:",
        "train_users="
        f"{len(training_inputs.split_interactions.train)}, "
        f"val_users={len(training_inputs.split_interactions.validation)}, "
        f"test_users={len(training_inputs.split_interactions.test)}, "
        f"item_universe={len(training_inputs.item_index.ids)}",
    )
    print(
        "[recommender.train_and_select] config:",
        f"embedding_dim={train_params.embedding_dim}, epochs={train_params.epochs}, "
        f"lr={train_params.learning_rate}, negatives={train_params.negative_samples}, "
        f"batch_size={train_params.batch_size}, l2={train_params.l2_reg}, "
        f"max_grad_norm={train_params.max_grad_norm}",
    )
    print(
        "[recommender.train_and_select] convergence:",
        f"early_stop_rounds={train_params.early_stop_rounds}, early_stop_k={train_params.early_stop_k}, "
        f"early_stop_tolerance={train_params.early_stop_tolerance}, temperature={train_params.temperature}",
    )


def run_train_and_select(config_path: str, output_dir_override=None) -> None:
    """What: Train recommender candidates, select one model, and write artifacts.
    Why: Provides the in-process training-stage entrypoint reused by the CLI, pipeline,
    and internal experiment utilities such as the tuning sweep.
    """
    # ===== Load Config And Validate Inputs =====
    runtime = load_recommender_runtime_config(config_path)
    cfg = runtime.cfg
    user_item_splits_path = runtime.user_item_splits_path
    user_index_path = runtime.user_index_path
    item_index_path = runtime.item_index_path
    output_dir = output_dir_override if output_dir_override is not None else runtime.artifacts_dir
    train_params = _load_train_params(cfg)
    _validate_train_inputs(user_item_splits_path, user_index_path, item_index_path, train_params)

    # ===== Load Split Rows And Build Shared Interaction Inputs =====
    training_inputs = _load_training_inputs(user_item_splits_path, user_index_path, item_index_path)
    split_interactions = training_inputs.split_interactions
    user_index = training_inputs.user_index
    item_index = training_inputs.item_index
    _log_training_context(training_inputs, train_params)

    # ===== Train All Candidate Models On The Same Split =====
    candidate_artifacts = train_candidate_models(
        split_interactions=split_interactions,
        user_index=user_index,
        item_index=item_index,
        train_params=train_params,
    )

    # ===== Run Offline Evaluation And Choose One Model =====
    selection = evaluate_and_select_model(
        user_ids=user_index.ids,
        train=split_interactions.train,
        validation=split_interactions.validation,
        test=split_interactions.test,
        user_to_idx=user_index.id_to_idx,
        item_to_idx=item_index.id_to_idx,
        candidate_artifacts=candidate_artifacts,
    )

    # ===== Persist Selected-Model Artifacts And Metrics =====
    _write_train_artifacts(
        output_dir,
        context=TrainArtifactContext(
            split_path=user_item_splits_path,
            evaluation_top_k=EVALUATION_TOP_K,
            user_ids=user_index.ids,
            item_ids=item_index.ids,
            user_to_idx=user_index.id_to_idx,
            item_to_idx=item_index.id_to_idx,
            train=split_interactions.train,
            validation=split_interactions.validation,
            test=split_interactions.test,
            model_config=asdict(train_params),
        ),
        candidate_artifacts=CandidateModelArtifactOutputs(
            artifacts_by_model=candidate_artifacts.artifacts_by_model,
        ),
        selection=SelectionArtifactOutputs(
            selected_model_name=selection.selected_model_name,
            validation_metrics=selection.validation_metrics,
            test_metrics=selection.test_metrics,
        ),
    )
    print(f"Wrote recommender train artifacts to: {output_dir}")


def main() -> None:
    """What: CLI entrypoint for recommender candidate training, evaluation, and selection.
    Why: Enables config-driven runs without exposing many CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Train recommender candidates and select one model.")
    parser.add_argument("--config", required=True, help="Recommender YAML config")
    args = parser.parse_args()
    run_train_and_select(config_path=args.config)


if __name__ == "__main__":
    main()
