"""Train purchase propensity scoring models and offline policy backtest."""

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from mle_marketplace_growth.purchase_propensity.helpers.data import (
    _apply_spend_cap,
    _feature_columns,
    _load_training_rows,
    _policy_scores,
    _quantile,
    _split_rows,
    _stable_ratio,
)
from mle_marketplace_growth.purchase_propensity.helpers.metrics import _expected_calibration_error, _safe_mape
from mle_marketplace_growth.purchase_propensity.helpers.modeling import _build_model, _fit_final_models, _fit_propensity_candidates, _fit_revenue_candidates

ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}
ALLOWED_FEATURE_LOOKBACK_WINDOWS = {60, 90, 120}


def main() -> None:
    # Parse CLI arguments.
    parser = argparse.ArgumentParser(description="Train propensity scoring models and backtest targeting policies.")
    parser.add_argument("--input-csv", default="data/gold/feature_store/purchase_propensity/propensity_train_dataset/as_of_date=2011-11-09/propensity_train_dataset.csv", help="Path to training dataset CSV")
    parser.add_argument("--output-dir", default="artifacts/purchase_propensity", help="Output directory for model and metrics")
    parser.add_argument("--spend-cap-quantile", type=float, default=0.99, help="Quantile cap for monetary_90d to reduce extreme-spend dominance (0 < q <= 1).")
    parser.add_argument("--prediction-window-days", type=int, default=30, help="Prediction horizon for purchase/revenue labels: 30, 60, 90")
    parser.add_argument("--feature-lookback-days", type=int, default=90, help="Feature lookback profile: 60, 90, 120")
    parser.add_argument("--force-propensity-model", choices=["logistic_regression", "xgboost"], default=None, help="Optional frozen propensity model from sensitivity validation.")
    parser.add_argument("--calibration-method", choices=["none", "sigmoid", "isotonic"], default="sigmoid", help="Probability calibration method applied on train folds before validation scoring.")
    args = parser.parse_args()

    # Validate key inputs and window settings.
    if not 0.0 < args.spend_cap_quantile <= 1.0: raise ValueError("--spend-cap-quantile must be in (0, 1]")
    if args.prediction_window_days not in ALLOWED_PREDICTION_WINDOWS: raise ValueError("--prediction-window-days must be one of: 30, 60, 90")
    if args.feature_lookback_days not in ALLOWED_FEATURE_LOOKBACK_WINDOWS: raise ValueError("--feature-lookback-days must be one of: 60, 90, 120")

    input_path = Path(args.input_csv)
    if not input_path.exists(): raise FileNotFoundError(f"Input CSV not found: {input_path}")

    feature_columns = _feature_columns(args.feature_lookback_days)
    spend_feature = f"monetary_{args.feature_lookback_days}d"
    purchase_label_column = f"label_purchase_{args.prediction_window_days}d"
    revenue_label_column = f"label_net_revenue_{args.prediction_window_days}d"

    # Load strict 10/1/1 panel split and construct capped feature matrices.
    rows = _load_training_rows(input_path, feature_columns=feature_columns, purchase_label_column=purchase_label_column, revenue_label_column=revenue_label_column)
    train_rows, validation_rows, test_rows, split_description = _split_rows(rows)
    if not train_rows or not validation_rows or not test_rows: raise ValueError("Train/validation/test split produced an empty subset.")

    train_labels = [int(row["purchase_label"]) for row in train_rows]
    if len(set(train_labels)) < 2: raise ValueError("Training labels contain only one class. Try an earlier as_of_date.")

    spend_cap_value = _quantile([float(row["features"][spend_feature]) for row in train_rows], args.spend_cap_quantile)
    _apply_spend_cap(train_rows, spend_feature, spend_cap_value)
    _apply_spend_cap(validation_rows, spend_feature, spend_cap_value)
    _apply_spend_cap(test_rows, spend_feature, spend_cap_value)

    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit([row["features"] for row in train_rows])
    train_matrix = vectorizer.transform([row["features"] for row in train_rows])
    validation_matrix = vectorizer.transform([row["features"] for row in validation_rows])
    validation_labels = [int(row["purchase_label"]) for row in validation_rows]

    # Train/compare propensity candidates on validation split.
    candidate_results = _fit_propensity_candidates(train_matrix, train_labels, validation_matrix, validation_labels, args.calibration_method)
    if args.force_propensity_model:
        forced = [row for row in candidate_results if row["model_name"] == args.force_propensity_model]
        if not forced: raise ValueError(f"Forced propensity model not found in candidates: {args.force_propensity_model}")
        best_result = forced[0]
    else:
        best_result = max(candidate_results, key=lambda result: result["average_precision"])

    selected_model_name = best_result["model_name"]
    validation_propensity_scores = best_result["propensity_scores"]
    selected_roc_auc, selected_average_precision = float(best_result["roc_auc"]), float(best_result["average_precision"])

    validation_positive_rate = sum(validation_labels) / len(validation_labels)
    validation_top_decile_count = max(1, int(0.1 * len(validation_labels)))
    validation_top_decile_indices = sorted(range(len(validation_rows)), key=lambda index: validation_propensity_scores[index], reverse=True)[:validation_top_decile_count]
    validation_top_decile_positive_rate = sum(validation_labels[index] for index in validation_top_decile_indices) / validation_top_decile_count
    validation_top_decile_lift = validation_top_decile_positive_rate / validation_positive_rate if validation_positive_rate > 0 else 0.0
    validation_brier_score = float(brier_score_loss(validation_labels, validation_propensity_scores))
    validation_ece_10_bin = _expected_calibration_error(validation_labels, validation_propensity_scores.tolist(), bins=10)

    # Train/compare conditional revenue models on validation positives.
    revenue_candidate_results, validation_revenue_predictions_by_candidate, selected_revenue_model_name, revenue_validation_quality = _fit_revenue_candidates(
        train_matrix,
        train_labels,
        train_rows,
        validation_matrix,
        validation_labels,
        validation_rows,
        purchase_label_column,
    )
    validation_revenue_predictions = validation_revenue_predictions_by_candidate[selected_revenue_model_name]

    # Refit selected models on development (train+validation) and score test.
    development_rows = train_rows + validation_rows
    development_labels = [int(row["purchase_label"]) for row in development_rows]
    final_vectorizer = DictVectorizer(sparse=True)
    final_vectorizer.fit([row["features"] for row in development_rows])
    development_matrix = final_vectorizer.transform([row["features"] for row in development_rows])
    test_matrix = final_vectorizer.transform([row["features"] for row in test_rows])

    selected_model, test_propensity_scores, revenue_model, revenue_model_name, revenue_fallback_value, test_predicted_conditional_revenue = _fit_final_models(
        development_matrix,
        development_labels,
        development_rows,
        test_matrix,
        test_rows,
        selected_model_name,
        selected_revenue_model_name,
        args.calibration_method,
    )

    test_labels = [int(row["purchase_label"]) for row in test_rows]
    test_base_positive_rate = sum(test_labels) / len(test_labels)
    test_top_decile_count = max(1, int(0.1 * len(test_labels)))
    test_top_decile_indices = sorted(range(len(test_rows)), key=lambda index: test_propensity_scores[index], reverse=True)[:test_top_decile_count]
    test_top_decile_positive_rate = sum(test_labels[index] for index in test_top_decile_indices) / test_top_decile_count
    test_top_decile_lift = test_top_decile_positive_rate / test_base_positive_rate if test_base_positive_rate > 0 else 0.0
    test_brier_score = float(brier_score_loss(test_labels, test_propensity_scores))
    test_ece_10_bin = _expected_calibration_error(test_labels, test_propensity_scores.tolist(), bins=10)

    positive_test_indices = [index for index, label in enumerate(test_labels) if label == 1]
    revenue_test_quality = {"evaluation_population": f"test_rows_with_{purchase_label_column}_equals_1", "row_count": 0}
    if positive_test_indices:
        test_revenue_actuals = [float(test_rows[index]["revenue_label"]) for index in positive_test_indices]
        test_revenue_predictions = [float(test_predicted_conditional_revenue[index]) for index in positive_test_indices]
        test_mape = _safe_mape(test_revenue_actuals, test_revenue_predictions)
        revenue_test_quality = {
            "evaluation_population": f"test_rows_with_{purchase_label_column}_equals_1",
            "row_count": len(positive_test_indices),
            "rmse": round(float(math.sqrt(mean_squared_error(test_revenue_actuals, test_revenue_predictions))), 6),
            "mae": round(float(mean_absolute_error(test_revenue_actuals, test_revenue_predictions)), 6),
            "mape": round(test_mape, 6) if test_mape is not None else None,
            "mape_note": "MAPE excludes rows with zero actual revenue.",
        }

    # Build policy scores for validation/test slices.
    test_expected_value_scores, test_random_scores, test_rfm_scores = _policy_scores(test_rows, test_propensity_scores.tolist(), test_predicted_conditional_revenue, args.feature_lookback_days)
    validation_expected_value_scores, validation_random_scores, validation_rfm_scores = _policy_scores(validation_rows, validation_propensity_scores.tolist(), validation_revenue_predictions, args.feature_lookback_days)

    # Persist model, metrics, and scored outputs.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "propensity_model.pkl"
    metrics_path = output_dir / "train_metrics.json"
    validation_scores_path = output_dir / "validation_predictions.csv"
    test_scores_path = output_dir / "test_predictions.csv"

    with model_path.open("wb") as file:
        pickle.dump(
            {
                "vectorizer": final_vectorizer,
                "propensity_model": selected_model,
                "propensity_model_name": selected_model_name,
                "revenue_model": revenue_model,
                "revenue_model_name": revenue_model_name,
                "revenue_fallback_value": round(float(revenue_fallback_value), 6),
                "feature_columns": feature_columns,
                "spend_cap_value": float(spend_cap_value),
                "spend_cap_quantile": float(args.spend_cap_quantile),
                "calibration_method": args.calibration_method,
                "prediction_window_days": args.prediction_window_days,
                "feature_lookback_days": args.feature_lookback_days,
                "target_definition": f"purchase_in_next_{args.prediction_window_days}_days",
                "expected_value_definition": f"propensity_score * predicted_conditional_revenue_{args.prediction_window_days}d",
            },
            file,
        )

    metrics_path.write_text(
        json.dumps(
            {
                "input_csv": str(input_path),
                "target_definition": purchase_label_column,
                "revenue_backtest_label": revenue_label_column,
                "prediction_window_days": args.prediction_window_days,
                "feature_lookback_days": args.feature_lookback_days,
                "validation_split_description": split_description,
                "spend_cap_quantile": args.spend_cap_quantile,
                "spend_cap_value": round(float(spend_cap_value), 6),
                "calibration_method": args.calibration_method,
                "revenue_model_name": revenue_model_name,
                "revenue_fallback_value": round(float(revenue_fallback_value), 6),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "test_rows": len(test_rows),
                "selected_model_name": selected_model_name,
                "selected_propensity_model_name": selected_model_name,
                "selected_revenue_model_name": revenue_model_name,
                "validation_quality": {
                    "roc_auc": round(selected_roc_auc, 6),
                    "average_precision": round(selected_average_precision, 6),
                    "top_decile_lift": round(float(validation_top_decile_lift), 6),
                    "top_decile_positive_rate": round(float(validation_top_decile_positive_rate), 6),
                    "base_positive_rate": round(float(validation_positive_rate), 6),
                    "brier_score": round(validation_brier_score, 6),
                    "ece_10_bin": round(validation_ece_10_bin, 6),
                },
                "test_quality": {
                    "roc_auc": round(float(roc_auc_score(test_labels, test_propensity_scores)), 6),
                    "average_precision": round(float(average_precision_score(test_labels, test_propensity_scores)), 6),
                    "top_decile_lift": round(float(test_top_decile_lift), 6),
                    "top_decile_positive_rate": round(float(test_top_decile_positive_rate), 6),
                    "base_positive_rate": round(float(test_base_positive_rate), 6),
                    "brier_score": round(test_brier_score, 6),
                    "ece_10_bin": round(test_ece_10_bin, 6),
                },
                "revenue_validation_quality": revenue_validation_quality,
                "revenue_test_quality": revenue_test_quality,
                "propensity_model_candidates": [
                    {
                        "model_name": result["model_name"],
                        "roc_auc": round(result["roc_auc"], 6),
                        "average_precision": round(result["average_precision"], 6),
                    }
                    for result in candidate_results
                ],
                "revenue_model_candidates": revenue_candidate_results,
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

    with validation_scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "as_of_date", purchase_label_column, revenue_label_column, "propensity_score", f"predicted_conditional_revenue_{args.prediction_window_days}d", "expected_value_score", "random_policy_score", "rfm_policy_score"])
        for index, (row, propensity, expected_value) in enumerate(zip(validation_rows, validation_propensity_scores, validation_expected_value_scores, strict=True)):
            writer.writerow([row["user_id"], row["as_of_date"], int(row["purchase_label"]), round(float(row["revenue_label"]), 6), round(float(propensity), 6), round(float(validation_revenue_predictions[index]), 6), round(float(expected_value), 6), round(float(validation_random_scores[index]), 6), round(float(validation_rfm_scores[index]), 6)])

    with test_scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "as_of_date", purchase_label_column, revenue_label_column, "propensity_score", f"predicted_conditional_revenue_{args.prediction_window_days}d", "expected_value_score", "random_policy_score", "rfm_policy_score"])
        for index, (row, propensity, expected_value) in enumerate(zip(test_rows, test_propensity_scores, test_expected_value_scores, strict=True)):
            writer.writerow([row["user_id"], row["as_of_date"], int(row["purchase_label"]), round(float(row["revenue_label"]), 6), round(float(propensity), 6), round(float(test_predicted_conditional_revenue[index]), 6), round(float(expected_value), 6), round(float(test_random_scores[index]), 6), round(float(test_rfm_scores[index]), 6)])

    print(f"Wrote model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote validation predictions: {validation_scores_path}")
    print(f"Wrote test predictions: {test_scores_path}")
    print(f"Selected model: {selected_model_name}")


if __name__ == "__main__":
    main()
