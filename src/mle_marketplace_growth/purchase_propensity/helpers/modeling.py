from __future__ import annotations

import math

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from mle_marketplace_growth.purchase_propensity.helpers.metrics import _safe_mape


def _build_model(model_name: str):
    if model_name == "logistic_regression":
        return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
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


def _fit_propensity_candidates(
    train_matrix,
    train_labels: list[int],
    validation_matrix,
    validation_labels: list[int],
    calibration_method: str,
) -> list[dict]:
    candidate_results = []
    for model_name in ["logistic_regression", "xgboost"]:
        model = _build_model(model_name)
        calibrated_model = CalibratedClassifierCV(estimator=model, method=calibration_method, cv=3) if calibration_method != "none" else model
        calibrated_model.fit(train_matrix, train_labels)
        propensity_scores = calibrated_model.predict_proba(validation_matrix)[:, 1]
        candidate_results.append(
            {
                "model_name": model_name,
                "model": calibrated_model,
                "propensity_scores": propensity_scores,
                "roc_auc": float(roc_auc_score(validation_labels, propensity_scores)),
                "average_precision": float(average_precision_score(validation_labels, propensity_scores)),
            }
        )
    return candidate_results


def _fit_revenue_candidates(
    train_matrix,
    train_labels: list[int],
    train_rows: list[dict],
    validation_matrix,
    validation_labels: list[int],
    validation_rows: list[dict],
    purchase_label_column: str,
) -> tuple[list[dict], dict[str, list[float]], str, dict]:
    positive_train_indices = [index for index, label in enumerate(train_labels) if label == 1]
    if not positive_train_indices: raise ValueError("No positive purchase rows for conditional revenue training.")
    positive_revenue_targets = [float(train_rows[index]["revenue_label"]) for index in positive_train_indices]
    validation_revenue_fallback = sum(positive_revenue_targets) / len(positive_revenue_targets)
    positive_validation_indices = [index for index, label in enumerate(validation_labels) if label == 1]

    revenue_candidate_results = []
    predictions_by_candidate: dict[str, list[float]] = {}
    for candidate_name in ["xgboost_regressor_conditional_revenue", "constant_mean_positive_revenue"]:
        if candidate_name == "xgboost_regressor_conditional_revenue" and len(positive_train_indices) >= 2:
            candidate_model = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            )
            candidate_model.fit(train_matrix[positive_train_indices], positive_revenue_targets)
            candidate_predictions = [max(0.0, float(value)) for value in candidate_model.predict(validation_matrix)]
        else:
            candidate_predictions = [max(0.0, validation_revenue_fallback) for _ in validation_rows]
        predictions_by_candidate[candidate_name] = candidate_predictions

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

    available = [row for row in revenue_candidate_results if row["rmse"] is not None]
    selected = min(available, key=lambda row: float(row["rmse"])) if available else revenue_candidate_results[0]
    selected_name = selected["model_name"]
    return (
        revenue_candidate_results,
        predictions_by_candidate,
        selected_name,
        {key: value for key, value in selected.items() if key != "model_name"},
    )


def _fit_final_models(
    development_matrix,
    development_labels: list[int],
    development_rows: list[dict],
    test_matrix,
    test_rows: list[dict],
    selected_model_name: str,
    selected_revenue_model_name: str,
    calibration_method: str,
):
    selected_model = _build_model(selected_model_name)
    if calibration_method != "none": selected_model = CalibratedClassifierCV(estimator=selected_model, method=calibration_method, cv=3)
    selected_model.fit(development_matrix, development_labels)
    test_propensity_scores = selected_model.predict_proba(test_matrix)[:, 1]

    development_positive_indices = [index for index, label in enumerate(development_labels) if label == 1]
    if not development_positive_indices: raise ValueError("No positive purchase rows for final conditional revenue training.")
    development_revenue_targets = [float(development_rows[index]["revenue_label"]) for index in development_positive_indices]
    revenue_fallback_value = sum(development_revenue_targets) / len(development_revenue_targets)
    revenue_model = None

    if selected_revenue_model_name == "xgboost_regressor_conditional_revenue" and len(development_positive_indices) >= 2:
        revenue_model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        revenue_model.fit(development_matrix[development_positive_indices], development_revenue_targets)

    if revenue_model is None:
        return selected_model, test_propensity_scores, None, "constant_mean_positive_revenue", revenue_fallback_value, [max(0.0, revenue_fallback_value) for _ in test_rows]
    return (
        selected_model,
        test_propensity_scores,
        revenue_model,
        "xgboost_regressor_conditional_revenue",
        revenue_fallback_value,
        [max(0.0, float(value)) for value in revenue_model.predict(test_matrix)],
    )
