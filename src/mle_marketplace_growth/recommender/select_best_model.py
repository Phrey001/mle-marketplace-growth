"""Winner selection for recommender candidate models.

Workflow Steps:
1) Read already-scored validation metric rows.
2) Select one winning model by validation Recall@K.
3) Return the selected model name for downstream reporting and artifact writing.
"""

from __future__ import annotations

from mle_marketplace_growth.recommender.constants import EVALUATION_TOP_K


def select_best_model(
    *,
    validation_metrics: list[dict],
) -> str:
    """What: Select the winning candidate model from offline validation metrics.
    Why: Freezes one selected model for downstream artifact writing and serving.
    """

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
    return selected_model_name
