"""Train purchase propensity scoring models and offline policy backtest.

This module runs a promotion-agnostic purchase propensity setup:
- Predict probability of purchase in the next 30 days.
- Predict conditional 30-day revenue for likely buyers.
- Rank users by expected value: purchase_probability * predicted_conditional_revenue_30d.
- Compare targeting policies on holdout outcomes.
- Cap extreme spend features (`monetary_90d`) at a high quantile before modeling and scoring.
- Calibrate predicted probabilities so scores are closer to observed purchase rates.

Scope:
- Offline backtest only; no causal promotional incrementality claim.
- True incremental impact still requires randomized online experimentation.

Why spend capping helps:
- Feature capping reduces sensitivity to extreme historical spend values.
- Without capping, a few very large historical spenders can destabilize both propensity and revenue models.
- Capping keeps model behavior more stable while preserving relative value signals.
"""

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}
ALLOWED_FEATURE_LOOKBACK_WINDOWS = {60, 90, 120}


# ===== Shared Utilities =====
def _quantile(values: list[float], q: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValueError("Cannot compute quantile on empty values.")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _feature_columns(feature_lookback_days: int) -> list[str]:
    return [
        "recency_days",
        "frequency_30d",
        f"frequency_{feature_lookback_days}d",
        "monetary_30d",
        f"monetary_{feature_lookback_days}d",
        f"avg_basket_value_{feature_lookback_days}d",
    ]


def _apply_spend_cap(rows: list[dict], spend_feature: str, spend_cap_value: float) -> None:
    for row in rows:
        row["features"][spend_feature] = min(row["features"][spend_feature], spend_cap_value)


def _stable_ratio(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    return value / float(0xFFFFFFFFFFFFFFFF)


# ===== Data Preparation =====
def _load_training_rows(
    input_path: Path,
    feature_columns: list[str],
    purchase_label_column: str,
    revenue_label_column: str,
) -> list[dict]:
    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            features = {feature: float(row[feature]) for feature in feature_columns}
            features["country"] = row["country"]
            purchase_label = float(row[purchase_label_column])
            revenue_label = float(row[revenue_label_column])
            rows.append(
                {
                    "user_id": row["user_id"],
                    "as_of_date": row["as_of_date"],
                    "features": features,
                    "purchase_label": purchase_label,
                    "revenue_label": revenue_label,
                }
            )
    if not rows:
        raise ValueError(f"No rows found in input dataset: {input_path}")
    return rows


# ===== Splits + Policy Metrics =====
def _split_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], str]:
    unique_dates = sorted({row["as_of_date"] for row in rows})
    if len(unique_dates) != 12:
        raise ValueError(
            "Strict split requires exactly 12 unique as_of_date snapshots "
            f"(got {len(unique_dates)})."
        )
    train_dates = set(unique_dates[:10])
    validation_dates = {unique_dates[10]}
    test_dates = {unique_dates[11]}
    train_rows = [row for row in rows if row["as_of_date"] in train_dates]
    validation_rows = [row for row in rows if row["as_of_date"] in validation_dates]
    test_rows = [row for row in rows if row["as_of_date"] in test_dates]
    split_desc = (
        f"out_of_time_10_1_1_train_dates={sorted(train_dates)};"
        f"validation_dates={sorted(validation_dates)};"
        f"test_dates={sorted(test_dates)}"
    )
    return train_rows, validation_rows, test_rows, split_desc


def _build_model(model_name: str):
    if model_name == "logistic_regression":
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _policy_scores(
    rows: list[dict],
    propensity_scores: list[float],
    predicted_conditional_revenue: list[float],
    feature_lookback_days: int,
) -> tuple[list[float], list[float], list[float]]:
    expected_value_scores = [
        float(score) * float(revenue)
        for score, revenue in zip(propensity_scores, predicted_conditional_revenue, strict=True)
    ]
    random_scores = [
        1.0 - _stable_ratio(f'{row["user_id"]}|{row["as_of_date"]}|policy_random')
        for row in rows
    ]
    freq_feature = f"frequency_{feature_lookback_days}d"
    monetary_feature = f"monetary_{feature_lookback_days}d"
    rfm_scores = [
        (1.0 / (1.0 + row["features"]["recency_days"]))
        + 0.5 * row["features"][freq_feature]
        + 0.01 * row["features"][monetary_feature]
        for row in rows
    ]
    return expected_value_scores, random_scores, rfm_scores


# ===== Calibration Utility =====
def _expected_calibration_error(labels: list[int], scores: list[float], bins: int = 10) -> float:
    total_rows = len(labels)
    weighted_gap = 0.0
    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        if idx == bins - 1:
            in_bin = [row_idx for row_idx, value in enumerate(scores) if lower <= value <= upper]
        else:
            in_bin = [row_idx for row_idx, value in enumerate(scores) if lower <= value < upper]
        if not in_bin:
            continue
        avg_score = sum(scores[row_idx] for row_idx in in_bin) / len(in_bin)
        avg_label = sum(labels[row_idx] for row_idx in in_bin) / len(in_bin)
        weighted_gap += abs(avg_score - avg_label) * (len(in_bin) / total_rows)
    return float(weighted_gap)


def _safe_mape(actuals: list[float], predictions: list[float]) -> float | None:
    non_zero_pairs = [
        (actual, prediction)
        for actual, prediction in zip(actuals, predictions, strict=True)
        if actual != 0
    ]
    if not non_zero_pairs:
        return None
    pct_errors = [abs((actual - prediction) / actual) for actual, prediction in non_zero_pairs]
    return float(sum(pct_errors) / len(pct_errors))


# ===== Entry Point =====
def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Train propensity scoring models and backtest targeting policies.")
    parser.add_argument("--input-csv", default="data/gold/feature_store/purchase_propensity/propensity_train_dataset/as_of_date=2011-11-09/propensity_train_dataset.csv", help="Path to training dataset CSV")
    parser.add_argument("--output-dir", default="artifacts/purchase_propensity", help="Output directory for model and metrics")
    parser.add_argument("--spend-cap-quantile", type=float, default=0.99, help="Quantile cap for monetary_90d to reduce extreme-spend dominance (0 < q <= 1).")
    parser.add_argument("--prediction-window-days", type=int, default=30, help="Prediction horizon for purchase/revenue labels: 30, 60, 90")
    parser.add_argument("--feature-lookback-days", type=int, default=90, help="Feature lookback profile: 60, 90, 120")
    parser.add_argument(
        "--force-propensity-model",
        choices=["logistic_regression", "xgboost"],
        default=None,
        help="Optional frozen propensity model from sensitivity validation.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration method applied on train folds before validation scoring.",
    )
    args = parser.parse_args()

    # ===== Input Checks =====
    if not 0.0 < args.spend_cap_quantile <= 1.0:
        raise ValueError("--spend-cap-quantile must be in (0, 1]")
    if args.prediction_window_days not in ALLOWED_PREDICTION_WINDOWS:
        raise ValueError("--prediction-window-days must be one of: 30, 60, 90")
    if args.feature_lookback_days not in ALLOWED_FEATURE_LOOKBACK_WINDOWS:
        raise ValueError("--feature-lookback-days must be one of: 60, 90, 120")

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # ===== Load Data + Split =====
    feature_columns = _feature_columns(args.feature_lookback_days)
    spend_feature = f"monetary_{args.feature_lookback_days}d"
    purchase_label_column = f"label_purchase_{args.prediction_window_days}d"
    revenue_label_column = f"label_net_revenue_{args.prediction_window_days}d"
    rows = _load_training_rows(
        input_path,
        feature_columns=feature_columns,
        purchase_label_column=purchase_label_column,
        revenue_label_column=revenue_label_column,
    )
    train_rows, validation_rows, test_rows, split_description = _split_rows(rows)
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Train/validation/test split produced an empty subset.")

    train_labels = [int(row["purchase_label"]) for row in train_rows]
    if len(set(train_labels)) < 2:
        raise ValueError("Training labels contain only one class. Try an earlier as_of_date.")

    spend_cap_value = _quantile(
        [float(row["features"][spend_feature]) for row in train_rows],
        args.spend_cap_quantile,
    )
    _apply_spend_cap(train_rows, spend_feature, spend_cap_value)
    _apply_spend_cap(validation_rows, spend_feature, spend_cap_value)
    _apply_spend_cap(test_rows, spend_feature, spend_cap_value)

    # ===== Train Model Candidates =====
    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit([row["features"] for row in train_rows])
    train_matrix = vectorizer.transform([row["features"] for row in train_rows])
    validation_matrix = vectorizer.transform([row["features"] for row in validation_rows])

    model_candidate_names = ["logistic_regression", "xgboost"]

    candidate_results = []
    validation_labels = [int(row["purchase_label"]) for row in validation_rows]
    for model_name in model_candidate_names:
        model = _build_model(model_name)
        calibrated_model = model
        if args.calibration_method != "none":
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                method=args.calibration_method,
                cv=3,
            )
        calibrated_model.fit(train_matrix, train_labels)
        propensity_scores = calibrated_model.predict_proba(validation_matrix)[:, 1]
        roc_auc = roc_auc_score(validation_labels, propensity_scores)
        avg_precision = average_precision_score(validation_labels, propensity_scores)
        candidate_results.append(
            {
                "model_name": model_name,
                "propensity_scores": propensity_scores,
                "roc_auc": float(roc_auc),
                "average_precision": float(avg_precision),
            }
        )

    if args.force_propensity_model:
        forced_candidates = [row for row in candidate_results if row["model_name"] == args.force_propensity_model]
        if not forced_candidates:
            raise ValueError(f"Forced propensity model not found in candidates: {args.force_propensity_model}")
        best_result = forced_candidates[0]
    else:
        best_result = max(candidate_results, key=lambda result: result["average_precision"])
    selected_model_name = best_result["model_name"]
    validation_propensity_scores = best_result["propensity_scores"]
    selected_roc_auc = float(best_result["roc_auc"])
    selected_average_precision = float(best_result["average_precision"])
    validation_positive_rate = sum(validation_labels) / len(validation_labels)
    validation_top_decile_count = max(1, int(0.1 * len(validation_labels)))
    validation_top_decile_indices = sorted(
        range(len(validation_rows)),
        key=lambda index: validation_propensity_scores[index],
        reverse=True,
    )[:validation_top_decile_count]
    validation_top_decile_positive_rate = (
        sum(validation_labels[index] for index in validation_top_decile_indices) / validation_top_decile_count
    )
    validation_top_decile_lift = (
        validation_top_decile_positive_rate / validation_positive_rate if validation_positive_rate > 0 else 0.0
    )
    validation_brier_score = float(brier_score_loss(validation_labels, validation_propensity_scores))
    validation_ece_10_bin = _expected_calibration_error(validation_labels, validation_propensity_scores.tolist(), bins=10)

    # Train validation-stage revenue model candidates on train positives only.
    positive_train_indices = [index for index, label in enumerate(train_labels) if label == 1]
    if not positive_train_indices:
        raise ValueError("No positive purchase rows for conditional revenue training.")
    positive_revenue_targets = [float(train_rows[index]["revenue_label"]) for index in positive_train_indices]
    validation_revenue_fallback = sum(positive_revenue_targets) / len(positive_revenue_targets)
    positive_validation_indices = [index for index, label in enumerate(validation_labels) if label == 1]
    revenue_candidate_results = []
    validation_revenue_predictions_by_candidate: dict[str, list[float]] = {}
    for candidate_name in ["xgboost_regressor_conditional_revenue", "constant_mean_positive_revenue"]:
        candidate_predictions: list[float]
        if candidate_name == "xgboost_regressor_conditional_revenue" and len(positive_train_indices) >= 2:
            positive_train_matrix = train_matrix[positive_train_indices]
            candidate_model = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            )
            candidate_model.fit(positive_train_matrix, positive_revenue_targets)
            candidate_predictions = [max(0.0, float(value)) for value in candidate_model.predict(validation_matrix)]
        else:
            candidate_predictions = [max(0.0, validation_revenue_fallback) for _ in validation_rows]
        validation_revenue_predictions_by_candidate[candidate_name] = candidate_predictions

        if not positive_validation_indices:
            revenue_candidate_results.append(
                {
                    "model_name": candidate_name,
                    "evaluation_population": f"validation_rows_with_{purchase_label_column}_equals_1",
                    "row_count": 0,
                    "rmse": None,
                    "mae": None,
                    "mape": None,
                    "status": "no_positive_validation_rows",
                }
            )
            continue

        revenue_actuals = [float(validation_rows[index]["revenue_label"]) for index in positive_validation_indices]
        revenue_predictions = [float(candidate_predictions[index]) for index in positive_validation_indices]
        mape = _safe_mape(revenue_actuals, revenue_predictions)
        revenue_candidate_results.append(
            {
                "model_name": candidate_name,
                "evaluation_population": f"validation_rows_with_{purchase_label_column}_equals_1",
                "row_count": len(positive_validation_indices),
                "rmse": round(float(math.sqrt(mean_squared_error(revenue_actuals, revenue_predictions))), 6),
                "mae": round(float(mean_absolute_error(revenue_actuals, revenue_predictions)), 6),
                "mape": round(mape, 6) if mape is not None else None,
                "mape_note": "MAPE excludes rows with zero actual revenue.",
                "status": "ok",
            }
        )

    available_revenue_candidates = [row for row in revenue_candidate_results if row["rmse"] is not None]
    selected_revenue_validation_result = (
        min(available_revenue_candidates, key=lambda row: float(row["rmse"]))
        if available_revenue_candidates
        else revenue_candidate_results[0]
    )
    selected_revenue_model_name = selected_revenue_validation_result["model_name"]
    validation_revenue_predictions = validation_revenue_predictions_by_candidate[selected_revenue_model_name]
    revenue_validation_quality = {
        key: value
        for key, value in selected_revenue_validation_result.items()
        if key != "model_name"
    }

    # ===== Final Refit on Train+Validation and Test Scoring =====
    development_rows = train_rows + validation_rows
    development_labels = [int(row["purchase_label"]) for row in development_rows]
    final_vectorizer = DictVectorizer(sparse=True)
    final_vectorizer.fit([row["features"] for row in development_rows])
    development_matrix = final_vectorizer.transform([row["features"] for row in development_rows])
    test_matrix = final_vectorizer.transform([row["features"] for row in test_rows])

    selected_model = _build_model(selected_model_name)
    if args.calibration_method != "none":
        selected_model = CalibratedClassifierCV(
            estimator=selected_model,
            method=args.calibration_method,
            cv=3,
        )
    selected_model.fit(development_matrix, development_labels)
    test_propensity_scores = selected_model.predict_proba(test_matrix)[:, 1]

    development_positive_indices = [index for index, label in enumerate(development_labels) if label == 1]
    if not development_positive_indices:
        raise ValueError("No positive purchase rows for final conditional revenue training.")
    development_revenue_targets = [
        float(development_rows[index]["revenue_label"]) for index in development_positive_indices
    ]
    revenue_fallback_value = sum(development_revenue_targets) / len(development_revenue_targets)
    revenue_model = None
    if selected_revenue_model_name == "xgboost_regressor_conditional_revenue" and len(development_positive_indices) >= 2:
        development_positive_matrix = development_matrix[development_positive_indices]
        revenue_model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        revenue_model.fit(development_positive_matrix, development_revenue_targets)
    if revenue_model is None:
        test_predicted_conditional_revenue = [max(0.0, revenue_fallback_value) for _ in test_rows]
        revenue_model_name = "constant_mean_positive_revenue"
    else:
        test_predicted_conditional_revenue = [
            max(0.0, float(value)) for value in revenue_model.predict(test_matrix)
        ]
        revenue_model_name = "xgboost_regressor_conditional_revenue"

    test_labels = [int(row["purchase_label"]) for row in test_rows]
    test_base_positive_rate = sum(test_labels) / len(test_labels)
    test_top_decile_count = max(1, int(0.1 * len(test_labels)))
    test_top_decile_indices = sorted(
        range(len(test_rows)),
        key=lambda index: test_propensity_scores[index],
        reverse=True,
    )[:test_top_decile_count]
    test_top_decile_positive_rate = sum(test_labels[index] for index in test_top_decile_indices) / test_top_decile_count
    test_top_decile_lift = test_top_decile_positive_rate / test_base_positive_rate if test_base_positive_rate > 0 else 0.0
    test_brier_score = float(brier_score_loss(test_labels, test_propensity_scores))
    test_ece_10_bin = _expected_calibration_error(test_labels, test_propensity_scores.tolist(), bins=10)

    positive_test_indices = [index for index, label in enumerate(test_labels) if label == 1]
    revenue_test_quality = {"evaluation_population": f"test_rows_with_{purchase_label_column}_equals_1", "row_count": 0}
    if positive_test_indices:
        test_revenue_actuals = [float(test_rows[index]["revenue_label"]) for index in positive_test_indices]
        test_revenue_predictions = [
            float(test_predicted_conditional_revenue[index]) for index in positive_test_indices
        ]
        test_mape = _safe_mape(test_revenue_actuals, test_revenue_predictions)
        revenue_test_quality = {
            "evaluation_population": f"test_rows_with_{purchase_label_column}_equals_1",
            "row_count": len(positive_test_indices),
            "rmse": round(float(math.sqrt(mean_squared_error(test_revenue_actuals, test_revenue_predictions))), 6),
            "mae": round(float(mean_absolute_error(test_revenue_actuals, test_revenue_predictions)), 6),
            "mape": round(test_mape, 6) if test_mape is not None else None,
            "mape_note": "MAPE excludes rows with zero actual revenue.",
        }

    test_expected_value_scores, test_random_scores, test_rfm_scores = _policy_scores(
        test_rows,
        test_propensity_scores.tolist(),
        test_predicted_conditional_revenue,
        args.feature_lookback_days,
    )
    validation_expected_value_scores, validation_random_scores, validation_rfm_scores = _policy_scores(
        validation_rows,
        validation_propensity_scores.tolist(),
        validation_revenue_predictions,
        args.feature_lookback_days,
    )

    # ===== Write Artifacts =====
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "propensity_model.pkl"
    metrics_path = output_dir / "train_metrics.json"
    validation_scores_path = output_dir / "validation_predictions.csv"
    test_scores_path = output_dir / "test_predictions.csv"

    model_bundle = {
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
    }
    with model_path.open("wb") as file:
        pickle.dump(model_bundle, file)

    metrics = {
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
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    with validation_scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "user_id",
                "as_of_date",
                purchase_label_column,
                revenue_label_column,
                "propensity_score",
                f"predicted_conditional_revenue_{args.prediction_window_days}d",
                "expected_value_score",
                "random_policy_score",
                "rfm_policy_score",
            ]
        )
        for index, (row, propensity, expected_value) in enumerate(
            zip(validation_rows, validation_propensity_scores, validation_expected_value_scores, strict=True)
        ):
            writer.writerow(
                [
                    row["user_id"],
                    row["as_of_date"],
                    int(row["purchase_label"]),
                    round(float(row["revenue_label"]), 6),
                    round(float(propensity), 6),
                    round(float(validation_revenue_predictions[index]), 6),
                    round(float(expected_value), 6),
                    round(float(validation_random_scores[index]), 6),
                    round(float(validation_rfm_scores[index]), 6),
                ]
            )

    with test_scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "user_id",
                "as_of_date",
                purchase_label_column,
                revenue_label_column,
                "propensity_score",
                f"predicted_conditional_revenue_{args.prediction_window_days}d",
                "expected_value_score",
                "random_policy_score",
                "rfm_policy_score",
            ]
        )
        for index, (row, propensity, expected_value) in enumerate(
            zip(test_rows, test_propensity_scores, test_expected_value_scores, strict=True)
        ):
            writer.writerow(
                [
                    row["user_id"],
                    row["as_of_date"],
                    int(row["purchase_label"]),
                    round(float(row["revenue_label"]), 6),
                    round(float(propensity), 6),
                    round(float(test_predicted_conditional_revenue[index]), 6),
                    round(float(expected_value), 6),
                    round(float(test_random_scores[index]), 6),
                    round(float(test_rfm_scores[index]), 6),
                ]
            )

    # ===== Run Summary =====
    print(f"Wrote model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote validation predictions: {validation_scores_path}")
    print(f"Wrote test predictions: {test_scores_path}")
    print(f"Selected model: {selected_model_name}")


if __name__ == "__main__":
    main()
