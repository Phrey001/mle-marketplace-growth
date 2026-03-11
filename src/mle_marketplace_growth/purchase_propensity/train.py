"""Train purchase propensity scoring models and offline policy backtest."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from mle_marketplace_growth.purchase_propensity.helpers.data import (
    _load_snapshot_rows,
    _quantile,
    _split_df_rows_10_1_1,
)
from mle_marketplace_growth.purchase_propensity.helpers.metrics import _propensity_quality_metrics
from mle_marketplace_growth.purchase_propensity.helpers.modeling import (
    _fit_test_conditional_revenue_model_wrapper,
    _fit_test_propensity_model_wrapper,
    _fit_validation_conditional_revenue_model_wrapper,
    _fit_validation_propensity_model_wrapper,
)

ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}
ALLOWED_FEATURE_LOOKBACK_WINDOWS = {60, 90, 120}
SPEND_CAP_QUANTILE = 0.99


def _apply_spend_cap(df: pd.DataFrame, spend_feature: str, spend_cap_value: float) -> None:
    """What: Cap one spend feature column in-place.
    Why: Limits outlier skew before model fitting and scoring.
    """
    df[spend_feature] = np.minimum(df[spend_feature].to_numpy(dtype=float), spend_cap_value)


def _fit_feature_encoding(train_feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """What: One-hot encode train features and return encoded frame plus column order.
    Why: Freezes the train schema so non-train splits can be aligned without leakage.
    """
    train_encoded_df = pd.get_dummies(train_feature_df, columns=["country"], dtype=float)
    return train_encoded_df, train_encoded_df.columns.tolist()


def _apply_feature_encoding(feature_df: pd.DataFrame, encoded_columns: list[str]) -> pd.DataFrame:
    """What: One-hot encode a split and align columns to the fitted train schema.
    Why: Keeps feature dimensions/order identical across train, validation, and test.
    """
    encoded_df = pd.get_dummies(feature_df, columns=["country"], dtype=float)
    return encoded_df.reindex(columns=encoded_columns, fill_value=0.0)


def _policy_scores(
    df: pd.DataFrame,
    propensity_scores: np.ndarray,
    predicted_conditional_revenue: np.ndarray,
    feature_lookback_days: int,
) -> tuple[list[float], list[float], list[float]]:
    """What: Build expected-value, random, and RFM policy scores per row.
    Why: Enables side-by-side offline policy comparison on the same population.
    """
    def _stable_ratio(key: str) -> float:
        """What: Deterministically map a key to a pseudo-random ratio in [0, 1)."""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)

    # A) Expected-value policy: propensity * predicted conditional revenue.
    expected_value_scores = (propensity_scores * predicted_conditional_revenue).tolist()
    # B) Random baseline policy: deterministic pseudo-random ranking per user/date.
    random_scores = [
        1.0 - _stable_ratio(f"{user_id}|{as_of_date}|policy_random")
        for user_id, as_of_date in zip(df["user_id"], df["as_of_date"], strict=True)
    ]
    # C) RFM heuristic policy: weighted recency + frequency + monetary blend.
    freq_feature, monetary_feature = f"frequency_{feature_lookback_days}d", f"monetary_{feature_lookback_days}d"
    rfm_scores = (
        (1.0 / (1.0 + df["recency_days"].to_numpy(dtype=float)))
        + 0.5 * df[freq_feature].to_numpy(dtype=float)
        + 0.01 * df[monetary_feature].to_numpy(dtype=float)
    ).tolist()
    return expected_value_scores, random_scores, rfm_scores


def main() -> None:
    """What: Train propensity and conditional revenue models, then write artifacts.
    Why: Produces deterministic offline backtest outputs for policy evaluation.
    """
    parser = argparse.ArgumentParser(description="Train propensity scoring models and backtest targeting policies.")
    parser.add_argument("--input-path", action="append", default=None, help="Path to training dataset parquet (repeat --input-path for multi-snapshot panel; action=append collects multiple values)")
    parser.add_argument("--output-dir", default="artifacts/purchase_propensity", help="Output directory for model and metrics")
    parser.add_argument("--prediction-window-days", type=int, default=30, help="Prediction horizon for purchase/revenue labels: 30, 60, 90")
    parser.add_argument("--feature-lookback-days", type=int, default=90, help="Feature lookback profile: 60, 90, 120")
    parser.add_argument("--force-propensity-model", choices=["logistic_regression", "xgboost"], required=True, help="Frozen propensity model from sensitivity or fixed config.")
    args = parser.parse_args()

    # 1) Validate key inputs and window settings.
    if args.prediction_window_days not in ALLOWED_PREDICTION_WINDOWS: raise ValueError("--prediction-window-days must be one of: 30, 60, 90")
    if args.feature_lookback_days not in ALLOWED_FEATURE_LOOKBACK_WINDOWS: raise ValueError("--feature-lookback-days must be one of: 60, 90, 120")

    # 2) Resolve input snapshots for the strict 12-month panel.
    if not args.input_path: raise ValueError("--input-path is required (repeat for multi-snapshot panel)")
    input_paths = [Path(path) for path in args.input_path]  # Each input path is one snapshot in the strict 12-month panel; convert to python list
    missing_inputs = [path for path in input_paths if not path.exists()]  # collect paths missing on disk if any
    if missing_inputs: raise FileNotFoundError(f"Input path not found: {missing_inputs[0]}")

    # 3) Define feature + label columns for this run.
    feature_columns = [
        # always include recency and short-term signals
        "recency_days",
        "frequency_30d",
        "monetary_30d",
        # dependent on lookback days
        f"frequency_{args.feature_lookback_days}d",
        f"monetary_{args.feature_lookback_days}d",
        f"avg_basket_value_{args.feature_lookback_days}d",
    ]
    spend_feature = f"monetary_{args.feature_lookback_days}d"  # feature already exist in feature_columns; for mutation in-place later
    purchase_label_column = f"label_purchase_{args.prediction_window_days}d"  # selected-horizon purchase label; used for loading and CSV schema
    revenue_label_column = f"label_net_revenue_{args.prediction_window_days}d"  # selected-horizon revenue label; used for loading revenue targets and CSV schema

    # 4) Load DataFrame, split into 10/1/1, and cap long-window spend.
    data_df = _load_snapshot_rows(
        input_paths,
        feature_columns=feature_columns,
        purchase_label_column=purchase_label_column,
        revenue_label_column=revenue_label_column,
    )
    train_df, validation_df, test_df, split_description = _split_df_rows_10_1_1(data_df)
    if train_df.empty or validation_df.empty or test_df.empty: raise ValueError("Train/validation/test split produced an empty subset.")

    train_purchase_labels = train_df["purchase_label"].to_numpy(dtype=int)
    if np.unique(train_purchase_labels).size < 2: raise ValueError("Training labels contain only one class. Try an earlier as_of_date.")

    spend_cap_value = _quantile(train_df[spend_feature].to_numpy(dtype=float), SPEND_CAP_QUANTILE)
    _apply_spend_cap(train_df, spend_feature, spend_cap_value)
    _apply_spend_cap(validation_df, spend_feature, spend_cap_value)
    _apply_spend_cap(test_df, spend_feature, spend_cap_value)

    # Feature preparation only (model training starts after this block).
    # Build raw feature tables (numeric + country) for this split.
    train_feature_df = train_df[[*feature_columns, "country"]].copy()
    validation_feature_df = validation_df[[*feature_columns, "country"]].copy()
    train_encoded_df, encoded_feature_columns = _fit_feature_encoding(train_feature_df)  # Fit one-hot schema on train only to avoid leakage.
    validation_encoded_df = _apply_feature_encoding(validation_feature_df, encoded_feature_columns)  # Apply same schema to non-train split, then convert to model matrices.
    train_matrix = train_encoded_df.to_numpy(dtype=float)
    validation_matrix = validation_encoded_df.to_numpy(dtype=float)
    validation_purchase_labels = validation_df["purchase_label"].to_numpy(dtype=int)
    train_conditional_revenue_labels = train_df["revenue_label"].to_numpy(dtype=float)
    validation_conditional_revenue_labels = validation_df["revenue_label"].to_numpy(dtype=float)

    # 5) Propensity model: train only the forced model for deterministic execution.
    selected_result = _fit_validation_propensity_model_wrapper(
        train_matrix,
        train_purchase_labels,
        validation_matrix,
        validation_purchase_labels,
        model_name=args.force_propensity_model,
        bins=10,
    )
    selected_model_name = str(selected_result["model_name"])
    validation_propensity_scores = selected_result["propensity_scores"]
    selected_roc_auc = float(selected_result["roc_auc"])
    selected_average_precision = float(selected_result["average_precision"])

    validation_quality_metrics = {
        "top_decile_lift": float(selected_result["top_decile_lift"]),
        "top_decile_positive_rate": float(selected_result["top_decile_positive_rate"]),
        "base_positive_rate": float(selected_result["base_positive_rate"]),
        "brier_score": float(selected_result["brier_score"]),
        "ece_10_bin": float(selected_result["ece_10_bin"]),
    }

    # 6) Revenue model: request XGB on validation, then freeze the effective path (XGB or fallback constant).
    # 6a) Fit/score conditional revenue on validation rows and freeze the effective model choice.
    _, frozen_revenue_model_name, _, validation_revenue_predictions, revenue_validation_quality = _fit_validation_conditional_revenue_model_wrapper(
        train_matrix,
        train_purchase_labels,
        train_conditional_revenue_labels,
        validation_matrix,
        validation_purchase_labels,
        validation_conditional_revenue_labels,
    )

    # 7) Final refit: selected propensity + revenue models on train+validation and score test.
    train_val_df = pd.concat([train_df, validation_df], ignore_index=True)
    train_val_purchase_labels = train_val_df["purchase_label"].to_numpy(dtype=int)
    train_val_conditional_revenue_labels = train_val_df["revenue_label"].to_numpy(dtype=float)
    # Refit one-hot column mapping on train+validation rows to avoid any label leakage from test rows.
    # Build raw feature tables (numeric + country) for the refit/test stage.
    train_val_feature_df = train_val_df[[*feature_columns, "country"]].copy()
    test_feature_df = test_df[[*feature_columns, "country"]].copy()
    train_val_encoded_df, train_val_encoded_feature_columns = _fit_feature_encoding(train_val_feature_df)  # Fit one-hot schema on refit rows only.
    test_encoded_df = _apply_feature_encoding(test_feature_df, train_val_encoded_feature_columns)  # Apply frozen refit schema to test rows, then convert to model matrices.
    train_val_matrix = train_val_encoded_df.to_numpy(dtype=float)
    test_matrix = test_encoded_df.to_numpy(dtype=float)

    selected_model, test_propensity_scores = _fit_test_propensity_model_wrapper(
        train_val_matrix,
        train_val_purchase_labels,
        test_matrix,
        selected_model_name,
    )
    # 7a) Reuse the frozen revenue model choice from validation; no model-type re-selection on test.
    revenue_model, revenue_model_name, revenue_fallback_value, test_predicted_conditional_revenue, revenue_test_quality = _fit_test_conditional_revenue_model_wrapper(
        train_val_matrix,
        train_val_purchase_labels,
        train_val_conditional_revenue_labels,
        test_matrix,
        test_df["purchase_label"].to_numpy(dtype=int),
        test_df["revenue_label"].to_numpy(dtype=float),
        frozen_revenue_model_name=frozen_revenue_model_name,
    )

    test_purchase_labels = test_df["purchase_label"].to_numpy(dtype=int)
    test_quality_metrics = _propensity_quality_metrics(test_purchase_labels, test_propensity_scores, bins=10)

    # 8) Build policy scores for validation/test slices.
    test_expected_value_scores, test_random_scores, test_rfm_scores = _policy_scores(test_df, test_propensity_scores, test_predicted_conditional_revenue, args.feature_lookback_days)
    validation_expected_value_scores, validation_random_scores, validation_rfm_scores = _policy_scores(validation_df, validation_propensity_scores, validation_revenue_predictions, args.feature_lookback_days)

    # 9) Persist model, metrics, and scored outputs.
    output_dir = Path(args.output_dir)  # Root folder for all training artifacts from this run.
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "propensity_model.pkl"  # Pickle artifact with fitted models + metadata used for downstream scoring.
    metrics_path = output_dir / "train_metrics.json"  # JSON summary of split sizes, model choices, and validation/test quality metrics.
    validation_scores_path = output_dir / "validation_predictions.csv"  # Row-level validation predictions for review/debug/policy replay.
    test_scores_path = output_dir / "test_predictions.csv"  # Row-level test predictions for final offline backtest analysis.

    # Model object artifact: what was trained and how to reproduce feature transforms.
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "propensity_model": selected_model,
                "propensity_model_name": selected_model_name,
                "revenue_model": revenue_model,
                "revenue_model_name": revenue_model_name,
                "revenue_fallback_value": round(float(revenue_fallback_value), 6),
                "feature_columns": feature_columns,
                "encoded_feature_columns": train_val_encoded_feature_columns,
                "spend_cap_value": float(spend_cap_value),
                "spend_cap_quantile": float(SPEND_CAP_QUANTILE),
                "calibration_method": "sigmoid",
                "prediction_window_days": args.prediction_window_days,
                "feature_lookback_days": args.feature_lookback_days,
                "target_definition": f"purchase_in_next_{args.prediction_window_days}_days",
                "expected_value_definition": f"propensity_score * predicted_conditional_revenue_{args.prediction_window_days}d",
            },
            file,
        )

    # Metrics artifact: compact run report used by pipeline selection and documentation.
    metrics_path.write_text(
        json.dumps(
            {
                "input_paths": [str(path) for path in input_paths],
                "target_definition": purchase_label_column,
                "revenue_backtest_label": revenue_label_column,
                "prediction_window_days": args.prediction_window_days,
                "feature_lookback_days": args.feature_lookback_days,
                "validation_split_description": split_description,
                "spend_cap_quantile": SPEND_CAP_QUANTILE,
                "spend_cap_value": round(float(spend_cap_value), 6),
                "calibration_method": "sigmoid",
                "revenue_model_name": revenue_model_name,
                "revenue_fallback_value": round(float(revenue_fallback_value), 6),
                "train_rows": len(train_df),
                "validation_rows": len(validation_df),
                "test_rows": len(test_df),
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
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Validation predictions artifact for slice-level inspection and QA.
    with validation_scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "as_of_date", purchase_label_column, revenue_label_column, "propensity_score", f"predicted_conditional_revenue_{args.prediction_window_days}d", "expected_value_score", "random_policy_score", "rfm_policy_score"])
        for index in range(len(validation_df)):
            writer.writerow(
                [
                    validation_df["user_id"].iat[index],
                    validation_df["as_of_date"].iat[index],
                    int(validation_df["purchase_label"].iat[index]),
                    round(float(validation_df["revenue_label"].iat[index]), 6),
                    round(float(validation_propensity_scores[index]), 6),
                    round(float(validation_revenue_predictions[index]), 6),
                    round(float(validation_expected_value_scores[index]), 6),
                    round(float(validation_random_scores[index]), 6),
                    round(float(validation_rfm_scores[index]), 6),
                ]
            )

    # Test predictions artifact for final offline comparison across targeting policies.
    with test_scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "as_of_date", purchase_label_column, revenue_label_column, "propensity_score", f"predicted_conditional_revenue_{args.prediction_window_days}d", "expected_value_score", "random_policy_score", "rfm_policy_score"])
        for index in range(len(test_df)):
            writer.writerow(
                [
                    test_df["user_id"].iat[index],
                    test_df["as_of_date"].iat[index],
                    int(test_df["purchase_label"].iat[index]),
                    round(float(test_df["revenue_label"].iat[index]), 6),
                    round(float(test_propensity_scores[index]), 6),
                    round(float(test_predicted_conditional_revenue[index]), 6),
                    round(float(test_expected_value_scores[index]), 6),
                    round(float(test_random_scores[index]), 6),
                    round(float(test_rfm_scores[index]), 6),
                ]
            )

    print(f"Wrote model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote validation predictions: {validation_scores_path}")
    print(f"Wrote test predictions: {test_scores_path}")
    print(f"Selected model: {selected_model_name}")


if __name__ == "__main__":
    main()
