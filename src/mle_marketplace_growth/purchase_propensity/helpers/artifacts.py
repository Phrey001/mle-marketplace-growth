"""Artifact writers/builders for purchase propensity training outputs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from mle_marketplace_growth.purchase_propensity.constants import SPEND_CAP_QUANTILE


def _dump_model_artifact(
    model_path: Path,
    selected_model,
    selected_model_name: str,
    revenue_model,
    revenue_model_name: str,
    revenue_fallback_value: float,
    feature_columns: list[str],
    encoded_feature_columns: list[str],
    spend_cap_value: float,
    prediction_window_days: int,
    feature_lookback_days: int,
) -> None:
    """What: Serialize fitted models and metadata to pickle.
    Why: Produces a reusable model artifact for downstream scoring/debug.
    """
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "propensity_model": selected_model,
                "propensity_model_name": selected_model_name,
                "revenue_model": revenue_model,
                "revenue_model_name": revenue_model_name,
                "revenue_fallback_value": round(float(revenue_fallback_value), 6),
                "feature_columns": feature_columns,
                "encoded_feature_columns": encoded_feature_columns,
                "spend_cap_value": float(spend_cap_value),
                "spend_cap_quantile": float(SPEND_CAP_QUANTILE),
                "calibration_method": "sigmoid",
                "prediction_window_days": prediction_window_days,
                "feature_lookback_days": feature_lookback_days,
                "target_definition": f"purchase_in_next_{prediction_window_days}_days",
                "expected_value_definition": f"propensity_score * predicted_conditional_revenue_{prediction_window_days}d",
            },
            file,
        )


def _build_train_metrics_payload(
    input_paths: list[Path],
    purchase_label_column: str,
    revenue_label_column: str,
    prediction_window_days: int,
    feature_lookback_days: int,
    split_description: str,
    spend_cap_value: float,
    revenue_model_name: str,
    revenue_fallback_value: float,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    selected_model_name: str,
    selected_roc_auc: float,
    selected_average_precision: float,
    validation_quality_metrics: dict,
    test_purchase_labels: np.ndarray,
    test_propensity_scores: np.ndarray,
    test_quality_metrics: dict,
    revenue_validation_quality: dict,
    revenue_test_quality: dict,
) -> dict:
    """What: Build train_metrics.json payload.
    Why: Keeps metrics schema construction outside train.py orchestration flow.
    """
    return {
        "input_paths": [str(path) for path in input_paths],
        "target_definition": purchase_label_column,
        "revenue_backtest_label": revenue_label_column,
        "prediction_window_days": prediction_window_days,
        "feature_lookback_days": feature_lookback_days,
        "validation_split_description": split_description,
        "spend_cap_quantile": SPEND_CAP_QUANTILE,
        "spend_cap_value": round(float(spend_cap_value), 6),
        "calibration_method": "sigmoid",
        "revenue_model_name": revenue_model_name,
        "revenue_fallback_value": round(float(revenue_fallback_value), 6),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "selected_model_name": selected_model_name,
        "selected_propensity_model_name": selected_model_name,
        "selected_revenue_model_name": revenue_model_name,
        "validation_quality": {
            "roc_auc": round(selected_roc_auc, 6),
            "average_precision": round(selected_average_precision, 6),
            "top_decile_lift": round(float(validation_quality_metrics["top_decile_lift"]), 6),
            "top_decile_positive_rate": round(float(validation_quality_metrics["top_decile_positive_rate"]), 6),
            "base_positive_rate": round(float(validation_quality_metrics["base_positive_rate"]), 6),
            "brier_score": round(float(validation_quality_metrics["brier_score"]), 6),
            "ece_10_bin": round(float(validation_quality_metrics["ece_10_bin"]), 6),
        },
        "test_quality": {
            "roc_auc": round(float(roc_auc_score(test_purchase_labels, test_propensity_scores)), 6),
            "average_precision": round(float(average_precision_score(test_purchase_labels, test_propensity_scores)), 6),
            "top_decile_lift": round(float(test_quality_metrics["top_decile_lift"]), 6),
            "top_decile_positive_rate": round(float(test_quality_metrics["top_decile_positive_rate"]), 6),
            "base_positive_rate": round(float(test_quality_metrics["base_positive_rate"]), 6),
            "brier_score": round(float(test_quality_metrics["brier_score"]), 6),
            "ece_10_bin": round(float(test_quality_metrics["ece_10_bin"]), 6),
        },
        "revenue_validation_quality": revenue_validation_quality,
        "revenue_test_quality": revenue_test_quality,
        "propensity_model_candidates": [
            {
                "model_name": selected_model_name,
                "roc_auc": round(selected_roc_auc, 6),
                "average_precision": round(selected_average_precision, 6),
                "top_decile_lift": round(float(validation_quality_metrics["top_decile_lift"]), 6),
                "top_decile_positive_rate": round(float(validation_quality_metrics["top_decile_positive_rate"]), 6),
                "base_positive_rate": round(float(validation_quality_metrics["base_positive_rate"]), 6),
                "brier_score": round(float(validation_quality_metrics["brier_score"]), 6),
                "ece_10_bin": round(float(validation_quality_metrics["ece_10_bin"]), 6),
            }
        ],
        "model_validation": {
            "selected_propensity_model_name": selected_model_name,
            "selected_revenue_model_name": revenue_model_name,
        },
        "scope": "offline_policy_budget_backtest_not_causal_promotional_incrementality",
    }


def _write_predictions_csv(
    output_path: Path,
    df: pd.DataFrame,
    purchase_label_column: str,
    revenue_label_column: str,
    prediction_window_days: int,
    propensity_scores: list[float] | np.ndarray,
    predicted_conditional_revenue: list[float] | np.ndarray,
    expected_value_scores: list[float] | np.ndarray,
    random_scores: list[float] | np.ndarray,
    rfm_scores: list[float] | np.ndarray,
) -> None:
    """What: Write row-level predictions CSV for one split.
    Why: Produces auditable artifacts for validation/test policy analysis.
    """
    predicted_revenue_column = f"predicted_conditional_revenue_{prediction_window_days}d"
    output_df = pd.DataFrame(
        {
            "user_id": df["user_id"].astype(str),
            "as_of_date": df["as_of_date"].astype(str),
            purchase_label_column: df["purchase_label"].astype(int),
            revenue_label_column: df["revenue_label"].astype(float),
            "propensity_score": np.asarray(propensity_scores, dtype=float),
            predicted_revenue_column: np.asarray(predicted_conditional_revenue, dtype=float),
            "expected_value_score": np.asarray(expected_value_scores, dtype=float),
            "random_policy_score": np.asarray(random_scores, dtype=float),
            "rfm_policy_score": np.asarray(rfm_scores, dtype=float),
        }
    )
    numeric_columns = [
        revenue_label_column,
        "propensity_score",
        predicted_revenue_column,
        "expected_value_score",
        "random_policy_score",
        "rfm_policy_score",
    ]
    output_df[numeric_columns] = output_df[numeric_columns].round(6)
    ordered_columns = [
        "user_id",
        "as_of_date",
        purchase_label_column,
        revenue_label_column,
        "propensity_score",
        predicted_revenue_column,
        "expected_value_score",
        "random_policy_score",
        "rfm_policy_score",
    ]
    output_df.to_csv(output_path, index=False, columns=ordered_columns)


def _write_metrics_artifact(metrics_path: Path, metrics_payload: dict) -> None:
    """What: Write train metrics payload to JSON file.
    Why: Centralizes JSON formatting/writing for train artifacts.
    """
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")


def _write_window_sensitivity_artifact(output_json_path: Path, output_payload: dict) -> None:
    """What: Write window sensitivity summary payload to JSON artifact.
    Why: Centralizes artifact-folder writes for sensitivity runs.
    """
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")


def _write_window_validation_dashboard(output_payload: dict, output_path: Path) -> None:
    """What: Render and save window-validation dashboard PNG artifact.
    Why: Produces a compact visual summary of sensitivity metric tradeoffs.
    """
    prediction_rows = output_payload.get("window_sensitivity", [])
    lookback_rows = output_payload.get("feature_window_validation", [])
    if not prediction_rows or not lookback_rows:
        return

    window_axis = [row["window_days"] for row in prediction_rows]
    best_window_metrics = [
        max(row["model_results"], key=lambda item: item["average_precision"]) for row in prediction_rows
    ]
    lookback_axis = [row["feature_lookback_days"] for row in lookback_rows]
    best_lookback_metrics = [
        max(row["model_results"], key=lambda item: item["average_precision"]) for row in lookback_rows
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metric_specs = [
        ("average_precision", "PR-AUC"),
        ("top_decile_lift", "Top-Decile Lift"),
        ("brier_score", "Brier Score"),
        ("ece_10_bin", "ECE (10-bin)"),
    ]
    for axis, (metric_key, title) in zip(axes.flat, metric_specs, strict=True):
        axis.plot(window_axis, [row[metric_key] for row in best_window_metrics], marker="o")
        axis.plot(lookback_axis, [row[metric_key] for row in best_lookback_metrics], marker="o")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("Window days")
    axes[0, 0].legend(["Prediction window", "Feature lookback"], loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote window validation dashboard: {output_path}")


def _write_batch_prediction_scores_artifact(
    output_path: Path,
    source_df: pd.DataFrame,
    propensity_scores: np.ndarray,
    conditional_revenue_scores: np.ndarray,
    expected_value_scores: np.ndarray,
    prediction_window_days: int,
) -> None:
    """What: Write batch prediction scores CSV artifact from scored arrays.
    Why: Centralizes prediction artifact schema/formatting in one helper module.
    """
    output_df = pd.DataFrame(
        {
            "user_id": source_df["user_id"].astype(str),
            "as_of_date": source_df["as_of_date"].astype(str),
            "propensity_score": np.asarray(propensity_scores, dtype=float),
            f"predicted_conditional_revenue_{prediction_window_days}d": np.asarray(conditional_revenue_scores, dtype=float),
            "expected_value_score": np.asarray(expected_value_scores, dtype=float),
        }
    )
    output_df[["propensity_score", f"predicted_conditional_revenue_{prediction_window_days}d", "expected_value_score"]] = output_df[
        ["propensity_score", f"predicted_conditional_revenue_{prediction_window_days}d", "expected_value_score"]
    ].round(6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
