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
from mle_marketplace_growth.recommender.helpers.artifacts import (
    CandidateModelArtifactOutputs,
    SelectionArtifactOutputs,
    TrainArtifactContext,
    _write_train_artifacts,
)
from mle_marketplace_growth.recommender.helpers.config import load_recommender_runtime_config
from mle_marketplace_growth.recommender.helpers.data import (
    _build_split_interactions,
    _load_item_index_df,
    _load_item_index,
    _load_user_index,
    _load_user_item_splits_df,
    _validate_split_chronology,
)
from mle_marketplace_growth.recommender.score_candidates import (
    score_mf_candidate,
    score_popularity_candidate,
    score_two_tower_candidate,
)
from mle_marketplace_growth.recommender.models.mf import MFTrainParams
from mle_marketplace_growth.recommender.models.two_tower import TwoTowerTrainParams
from mle_marketplace_growth.recommender.select_best_model import select_best_model
from mle_marketplace_growth.recommender.train_candidates import (
    train_mf_artifacts,
    train_popularity_artifacts,
    train_two_tower_artifacts,
)


def _load_mf_train_params(cfg: dict) -> MFTrainParams:
    """What: Parse MF-only hyperparameters from config into the MF model contract.
    Why: Keeps MF config ownership model-specific instead of mixing it with other model knobs.
    """
    return MFTrainParams(
        mf_components=int(cfg_required(cfg, "mf_components")),
        mf_n_iter=int(cfg_required(cfg, "mf_n_iter")),
        mf_weighting=str(cfg_required(cfg, "mf_weighting")),
    )


def _load_two_tower_train_params(cfg: dict) -> TwoTowerTrainParams:
    """What: Parse two-tower-only hyperparameters from config into the two-tower model contract.
    Why: Keeps two-tower config ownership model-specific instead of mixing it with MF knobs.
    """
    return TwoTowerTrainParams(
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
    )


def _validate_train_inputs(
    user_item_splits_path,
    user_index_path,
    item_index_path,
    mf_params: MFTrainParams,
    two_tower_params: TwoTowerTrainParams,
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
    if two_tower_params.embedding_dim < 2:
        raise ValueError("embedding_dim must be >= 2")
    if two_tower_params.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if two_tower_params.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if two_tower_params.negative_samples < 0:
        raise ValueError("negative_samples must be >= 0")
    if two_tower_params.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if two_tower_params.l2_reg < 0:
        raise ValueError("l2_reg must be >= 0")
    if two_tower_params.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be >= 0")
    if two_tower_params.early_stop_rounds < 0:
        raise ValueError("early_stop_rounds must be >= 0")
    if two_tower_params.early_stop_k < 1:
        raise ValueError("early_stop_k must be >= 1")
    if two_tower_params.early_stop_tolerance < 0:
        raise ValueError("early_stop_tolerance must be >= 0")
    if two_tower_params.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if mf_params.mf_components < 2:
        raise ValueError("mf_components must be >= 2")
    if mf_params.mf_n_iter < 1:
        raise ValueError("mf_n_iter must be >= 1")
    if mf_params.mf_weighting not in ALLOWED_MF_WEIGHTINGS:
        raise ValueError(f"mf_weighting must be one of {list(ALLOWED_MF_WEIGHTINGS)}")


def _load_training_inputs(user_item_splits_path, user_index_path, item_index_path):
    """What: Load and normalize all shared inputs needed by candidate training/evaluation.
    Why: Keeps the top-level stage entrypoint focused on orchestration rather than setup details.
    """
    user_item_splits_df = _load_user_item_splits_df(user_item_splits_path)
    _validate_split_chronology(user_item_splits_df)
    split_interactions = _build_split_interactions(user_item_splits_df)
    user_index = _load_user_index(user_index_path)
    item_index_df = _load_item_index_df(item_index_path)
    item_index = _load_item_index(item_index_path)
    return user_item_splits_df, split_interactions, user_index, item_index, item_index_df


def _log_training_context(
    split_interactions,
    user_index,
    item_index,
    mf_params: MFTrainParams,
    two_tower_params: TwoTowerTrainParams,
) -> None:
    """What: Print concise run context before candidate training starts.
    Why: Makes the stage easier to inspect without scattering print statements across the main flow.
    """
    print(
        "[recommender.train_and_select] loaded splits:",
        "train_users="
        f"{len(split_interactions.train)}, "
        f"val_users={len(split_interactions.validation)}, "
        f"test_users={len(split_interactions.test)}, "
        f"item_universe={len(item_index.ids)}",
    )
    print(
        "[recommender.train_and_select] config:",
        f"embedding_dim={two_tower_params.embedding_dim}, epochs={two_tower_params.epochs}, "
        f"lr={two_tower_params.learning_rate}, negatives={two_tower_params.negative_samples}, "
        f"batch_size={two_tower_params.batch_size}, l2={two_tower_params.l2_reg}, "
        f"max_grad_norm={two_tower_params.max_grad_norm}",
    )
    print(
        "[recommender.train_and_select] convergence:",
        f"early_stop_rounds={two_tower_params.early_stop_rounds}, early_stop_k={two_tower_params.early_stop_k}, "
        f"early_stop_tolerance={two_tower_params.early_stop_tolerance}, temperature={two_tower_params.temperature}",
    )
    print(
        "[recommender.train_and_select] mf:",
        f"components={mf_params.mf_components}, n_iter={mf_params.mf_n_iter}, "
        f"weighting={mf_params.mf_weighting}",
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
    mf_params = _load_mf_train_params(cfg)
    two_tower_params = _load_two_tower_train_params(cfg)
    _validate_train_inputs(user_item_splits_path, user_index_path, item_index_path, mf_params, two_tower_params)

    # ===== Load Split Rows And Build Shared Interaction Inputs =====
    user_item_splits_df, split_interactions, user_index, item_index, item_index_df = _load_training_inputs(
        user_item_splits_path,
        user_index_path,
        item_index_path,
    )
    _log_training_context(split_interactions, user_index, item_index, mf_params, two_tower_params)

    # ===== Train All Candidate Models On The Same Split =====
    popularity_artifacts = train_popularity_artifacts(
        user_item_splits_df=user_item_splits_df,
        item_index_df=item_index_df,
    )
    mf_artifacts = train_mf_artifacts(
        split_interactions=split_interactions,
        user_index=user_index,
        item_index=item_index,
        mf_params=mf_params,
    )
    two_tower_artifacts = train_two_tower_artifacts(
        split_interactions=split_interactions,
        user_index=user_index,
        item_index=item_index,
        two_tower_params=two_tower_params,
    )

    # ===== Score Candidates Offline And Choose One Model =====
    popularity_validation_metrics = score_popularity_candidate(
        users=user_index.ids,
        train_rows=split_interactions.train,
        target_rows=split_interactions.validation,
        user_to_idx=user_index.id_to_idx,
        item_to_idx=item_index.id_to_idx,
        popularity_artifacts=popularity_artifacts,
    )
    mf_validation_metrics = score_mf_candidate(
        users=user_index.ids,
        train_rows=split_interactions.train,
        target_rows=split_interactions.validation,
        user_to_idx=user_index.id_to_idx,
        item_to_idx=item_index.id_to_idx,
        mf_artifacts=mf_artifacts,
    )
    two_tower_validation_metrics = score_two_tower_candidate(
        users=user_index.ids,
        train_rows=split_interactions.train,
        target_rows=split_interactions.validation,
        user_to_idx=user_index.id_to_idx,
        item_to_idx=item_index.id_to_idx,
        two_tower_artifacts=two_tower_artifacts,
    )
    popularity_test_metrics = score_popularity_candidate(
        users=user_index.ids,
        train_rows=split_interactions.train,
        target_rows=split_interactions.test,
        user_to_idx=user_index.id_to_idx,
        item_to_idx=item_index.id_to_idx,
        popularity_artifacts=popularity_artifacts,
    )
    mf_test_metrics = score_mf_candidate(
        users=user_index.ids,
        train_rows=split_interactions.train,
        target_rows=split_interactions.test,
        user_to_idx=user_index.id_to_idx,
        item_to_idx=item_index.id_to_idx,
        mf_artifacts=mf_artifacts,
    )
    two_tower_test_metrics = score_two_tower_candidate(
        users=user_index.ids,
        train_rows=split_interactions.train,
        target_rows=split_interactions.test,
        user_to_idx=user_index.id_to_idx,
        item_to_idx=item_index.id_to_idx,
        two_tower_artifacts=two_tower_artifacts,
    )
    validation_metrics = [
        popularity_validation_metrics,
        mf_validation_metrics,
        two_tower_validation_metrics,
    ]
    test_metrics = [
        popularity_test_metrics,
        mf_test_metrics,
        two_tower_test_metrics,
    ]
    selected_model_name = select_best_model(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )

    # ===== Persist Selected-Model Artifacts And Metrics =====
    _write_train_artifacts(
        output_dir,
        context=TrainArtifactContext(
            evaluation_top_k=EVALUATION_TOP_K,
            user_ids=user_index.ids,
            item_ids=item_index.ids,
            user_to_idx=user_index.id_to_idx,
            item_to_idx=item_index.id_to_idx,
            train=split_interactions.train,
            model_config={
                "mf": asdict(mf_params),
                "two_tower": asdict(two_tower_params),
            },
        ),
        candidate_artifacts=CandidateModelArtifactOutputs(
            artifacts_by_model={
                "popularity": popularity_artifacts,
                "mf": mf_artifacts,
                "two_tower": two_tower_artifacts,
            },
        ),
        selection=SelectionArtifactOutputs(
            selected_model_name=selected_model_name,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
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
