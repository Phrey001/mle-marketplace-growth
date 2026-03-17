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
from dataclasses import asdict, dataclass

import numpy as np

from mle_marketplace_growth.helpers import cfg_required
from mle_marketplace_growth.recommender.constants import ALLOWED_MF_WEIGHTINGS, EVALUATION_TOP_K, MODEL_NAMES
from mle_marketplace_growth.recommender.helpers.artifacts import (
    CandidateModelArtifactOutputs,
    SelectionArtifactOutputs,
    TrainArtifactContext,
    _write_train_artifacts,
)
from mle_marketplace_growth.recommender.helpers.config import load_recommender_runtime_config
from mle_marketplace_growth.recommender.helpers.data import (
    _build_interactions,
    _load_entity_index,
    _load_split_rows,
    _validate_split_chronology,
)
from mle_marketplace_growth.recommender.helpers.metrics import _evaluate_ranked_items, _top_k_indices, _user_eval_pool
from mle_marketplace_growth.recommender.models.mf import MFScorer, train_mf_candidate
from mle_marketplace_growth.recommender.models.popularity import PopularityScorer, train_popularity_candidate
from mle_marketplace_growth.recommender.models.two_tower import TwoTowerScorer, train_two_tower_candidate


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
class CandidateModelArtifacts:
    artifacts_by_model: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class ModelSelectionResult:
    validation_metrics: list[dict]
    test_metrics: list[dict]
    selected_model_name: str


MODEL_SCORER_REGISTRY = {
    "popularity": lambda *, model_artifacts: PopularityScorer(scores=model_artifacts["scores"]),
    "mf": lambda *, model_artifacts: MFScorer(
        user_embeddings=model_artifacts["user_embeddings"],
        item_embeddings=model_artifacts["item_embeddings"],
    ),
    "two_tower": lambda *, model_artifacts: TwoTowerScorer(
        user_embeddings=model_artifacts["user_embeddings"],
        item_embeddings=model_artifacts["item_embeddings"],
    ),
}


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
    split_path,
    user_index_path,
    item_index_path,
    train_params: RecommenderTrainParams,
) -> None:
    """What: Validate filesystem inputs and semantic training preconditions after config parsing.
    Why: Fails fast with focused error messages before candidate training starts.
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


def _train_candidate_models(
    train: dict[str, set[str]],
    validation: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    train_params: RecommenderTrainParams,
) -> CandidateModelArtifacts:
    """What: Train popularity, MF, and two-tower candidates through dedicated model modules.
    Why: This stage intentionally trains all supported model families in one run so they are compared fairly on the same data split.
    """
    popularity_scores = train_popularity_candidate(train, item_to_idx)
    mf_user_embeddings, mf_item_embeddings = train_mf_candidate(
        train,
        user_to_idx,
        item_to_idx,
        mf_components=train_params.mf_components,
        mf_n_iter=train_params.mf_n_iter,
        mf_weighting=train_params.mf_weighting,
    )
    tt_user_embeddings, tt_item_embeddings = train_two_tower_candidate(
        train,
        validation,
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
        early_stop_k=train_params.early_stop_k,
        early_stop_tolerance=train_params.early_stop_tolerance,
        temperature=train_params.temperature,
    )
    return CandidateModelArtifacts(
        artifacts_by_model={
            "popularity": {"scores": popularity_scores},
            "mf": {
                "user_embeddings": mf_user_embeddings,
                "item_embeddings": mf_item_embeddings,
            },
            "two_tower": {
                "user_embeddings": tt_user_embeddings,
                "item_embeddings": tt_item_embeddings,
            },
        },
    )


def _evaluate_and_select_model(
    user_ids: list[str],
    train: dict[str, set[str]],
    validation: dict[str, set[str]],
    test: dict[str, set[str]],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    candidate_artifacts: CandidateModelArtifacts,
) -> ModelSelectionResult:
    """What: Evaluate all candidates and select best by validation Recall@K.
    Why: Freezes one selected model for downstream artifact writing and serving.
    """
    def _evaluate_model(
        *,
        model_name: str,
        users: list[str],
        train_rows: dict[str, set[str]],
        split_rows: dict[str, set[str]],
    ) -> dict:
        """What: Evaluate one model family on one split at the configured K cutoff.
        Why: Produces comparable offline retrieval metrics for model selection inside this stage.
        """
        def _eligible_user_pool(user_id: str) -> tuple[list[int], set[int], int] | None:
            if user_id not in split_rows or user_id not in train_rows or user_id not in user_to_idx:
                return None
            pool = _user_eval_pool(train_rows[user_id], split_rows[user_id], item_to_idx)
            if pool is None:
                return None
            candidate_item_indices, ground_truth_indices = pool
            return candidate_item_indices, ground_truth_indices, user_to_idx[user_id]

        scorer_builder = MODEL_SCORER_REGISTRY.get(model_name)
        if scorer_builder is None:
            raise ValueError(f"Unsupported model: {model_name}")
        model_artifacts = candidate_artifacts.artifacts_by_model.get(model_name)
        if model_artifacts is None:
            raise ValueError(f"Missing candidate artifacts for model: {model_name}")
        scorer = scorer_builder(model_artifacts=model_artifacts)
        metric_sums = {"recall": 0.0, "ndcg": 0.0, "hit_rate": 0.0}
        eligible_users = 0
        item_count = len(item_to_idx)

        for user_id in users:
            eligible = _eligible_user_pool(user_id)
            if eligible is None:
                continue
            candidate_item_indices, ground_truth_indices, user_index = eligible
            eligible_users += 1
            candidate_scores = scorer.score_candidate_indices(user_index, candidate_item_indices)
            effective_k = min(EVALUATION_TOP_K, len(candidate_item_indices), item_count)
            top_local_indices = _top_k_indices(candidate_scores, effective_k)
            ranked_item_indices = [candidate_item_indices[idx] for idx in top_local_indices]
            row_metrics = _evaluate_ranked_items(ranked_item_indices, ground_truth_indices, effective_k)
            metric_sums["recall"] += row_metrics["recall"]
            metric_sums["hit_rate"] += row_metrics["hit_rate"]
            metric_sums["ndcg"] += row_metrics["ndcg"]

        if eligible_users == 0:
            metrics = {
                f"Recall@{EVALUATION_TOP_K}": 0.0,
                f"NDCG@{EVALUATION_TOP_K}": 0.0,
                f"HitRate@{EVALUATION_TOP_K}": 0.0,
            }
        else:
            metrics = {
                f"Recall@{EVALUATION_TOP_K}": round(metric_sums["recall"] / eligible_users, 6),
                f"NDCG@{EVALUATION_TOP_K}": round(metric_sums["ndcg"] / eligible_users, 6),
                f"HitRate@{EVALUATION_TOP_K}": round(metric_sums["hit_rate"] / eligible_users, 6),
            }
        return {"model_name": model_name, "eligible_users": eligible_users, "metrics": metrics}

    model_names = list(MODEL_NAMES)
    metrics_by_split = {
        split_name: [
            _evaluate_model(
                model_name=name,
                users=user_ids,
                train_rows=train,
                split_rows=split_rows,
            )
            for name in model_names
        ]
        for split_name, split_rows in [("validation", validation), ("test", test)]
    }
    validation_metrics, test_metrics = metrics_by_split["validation"], metrics_by_split["test"]
    for result in validation_metrics:
        print(
            "[recommender.train_and_select] validation:",
            result["model_name"],
            f"Recall@{EVALUATION_TOP_K}={result['metrics'].get(f'Recall@{EVALUATION_TOP_K}', 0.0):.6f}",
        )
    selected_model_name = max(
        validation_metrics,
        key=lambda result: result["metrics"].get(f"Recall@{EVALUATION_TOP_K}", 0.0),
    )["model_name"]
    print(f"[recommender.train_and_select] selected_model={selected_model_name} by Recall@{EVALUATION_TOP_K}")
    selected_test_row = next((result for result in test_metrics if result["model_name"] == selected_model_name), None)
    if selected_test_row is not None:
        print(
            "[recommender.train_and_select] test:",
            selected_model_name,
            f"Recall@{EVALUATION_TOP_K}={selected_test_row['metrics'].get(f'Recall@{EVALUATION_TOP_K}', 0.0):.6f}",
        )
    return ModelSelectionResult(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        selected_model_name=selected_model_name,
    )


def run_train_and_select(config_path: str, output_dir_override=None) -> None:
    """What: Train recommender candidates, select one model, and write artifacts.
    Why: Provides the in-process training-stage entrypoint reused by the CLI, pipeline,
    and internal experiment utilities such as the tuning sweep.
    """
    runtime = load_recommender_runtime_config(config_path)
    cfg = runtime.cfg
    split_path = runtime.splits_path
    user_index_path = runtime.user_index_path
    item_index_path = runtime.item_index_path
    output_dir = output_dir_override if output_dir_override is not None else runtime.artifacts_dir
    train_params = _load_train_params(cfg)
    _validate_train_inputs(split_path, user_index_path, item_index_path, train_params)

    rows_df = _load_split_rows(split_path)
    _validate_split_chronology(rows_df)
    train, validation, test = _build_interactions(rows_df)
    user_ids, user_to_idx = _load_entity_index(user_index_path, id_col="user_id", idx_col="user_idx")
    item_ids, item_to_idx = _load_entity_index(item_index_path, id_col="item_id", idx_col="item_idx")
    print(
        "[recommender.train_and_select] loaded splits:",
        f"train_users={len(train)}, val_users={len(validation)}, test_users={len(test)}, item_universe={len(item_ids)}",
    )
    print(
        "[recommender.train_and_select] config:",
        f"embedding_dim={train_params.embedding_dim}, epochs={train_params.epochs}, lr={train_params.learning_rate}, negatives={train_params.negative_samples}, batch_size={train_params.batch_size}, l2={train_params.l2_reg}, max_grad_norm={train_params.max_grad_norm}",
    )
    print(
        "[recommender.train_and_select] convergence:",
        f"early_stop_rounds={train_params.early_stop_rounds}, early_stop_k={train_params.early_stop_k}, early_stop_tolerance={train_params.early_stop_tolerance}, temperature={train_params.temperature}",
    )

    candidate_artifacts = _train_candidate_models(
        train,
        validation,
        user_to_idx,
        item_to_idx,
        train_params,
    )
    selection = _evaluate_and_select_model(
        user_ids,
        train,
        validation,
        test,
        user_to_idx,
        item_to_idx,
        candidate_artifacts,
    )
    _write_train_artifacts(
        output_dir,
        context=TrainArtifactContext(
            split_path=split_path,
            evaluation_top_k=EVALUATION_TOP_K,
            user_ids=user_ids,
            item_ids=item_ids,
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx,
            train=train,
            validation=validation,
            test=test,
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
