"""Offline evaluation and model selection for recommender candidate artifacts.

Workflow Steps:
1) Score each trained candidate on shared validation and test user pools.
2) Aggregate Recall/NDCG/HitRate metrics at the fixed evaluation cutoff.
3) Select one winning model by validation Recall@K.
"""

from __future__ import annotations

from mle_marketplace_growth.recommender.constants import EVALUATION_TOP_K, MODEL_NAMES
from mle_marketplace_growth.recommender.contracts import CandidateModelArtifacts, ModelSelectionResult
from mle_marketplace_growth.recommender.helpers.metrics import _evaluate_ranked_items, _top_k_indices, _user_eval_pool
from mle_marketplace_growth.recommender.models.mf import MFScorer
from mle_marketplace_growth.recommender.models.popularity import PopularityScorer
from mle_marketplace_growth.recommender.models.two_tower import TwoTowerScorer

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


def evaluate_and_select_model(
    *,
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
            "[recommender.select_best_model] validation:",
            result["model_name"],
            f"Recall@{EVALUATION_TOP_K}={result['metrics'].get(f'Recall@{EVALUATION_TOP_K}', 0.0):.6f}",
        )

    selected_model_name = max(
        validation_metrics,
        key=lambda result: result["metrics"].get(f"Recall@{EVALUATION_TOP_K}", 0.0),
    )["model_name"]
    print(f"[recommender.select_best_model] selected_model={selected_model_name} by Recall@{EVALUATION_TOP_K}")
    selected_test_row = next((result for result in test_metrics if result["model_name"] == selected_model_name), None)
    if selected_test_row is not None:
        print(
            "[recommender.select_best_model] test:",
            selected_model_name,
            f"Recall@{EVALUATION_TOP_K}={selected_test_row['metrics'].get(f'Recall@{EVALUATION_TOP_K}', 0.0):.6f}",
        )
    return ModelSelectionResult(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        selected_model_name=selected_model_name,
    )
