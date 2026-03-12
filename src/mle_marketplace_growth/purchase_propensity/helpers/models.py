from __future__ import annotations

"""Modeling helpers for purchase propensity training.

These helpers build/fit candidate models for:
- purchase propensity (classification)
- conditional revenue (regression on positives)
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from mle_marketplace_growth.purchase_propensity.helpers.metrics import _propensity_quality_metrics, _safe_mape


def _build_propensity_model(model_name: str):
    """What: Return an untrained propensity classifier by model name.
    Why: Centralizes model construction so training/evaluation use the same spec.
    """
    if model_name == "logistic_regression":
        # Standardize numeric features and balance classes for sparse inputs.
        return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    if model_name == "xgboost":
        # Gradient-boosted trees for non-linear patterns; tuned for moderate depth.
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


def _fit_and_score_propensity_model(
    X_fit,
    y_fit: np.ndarray,
    X_pred,
    model_name: str,
) -> tuple[object, np.ndarray]:
    """What: Fit a calibrated propensity model and produce predicted probabilities.
    Why: Provides a shared fit/score path used in both validation and final refit stages.

    Note: Calibration is important because the predicted probability is used
    directly in expected-value ranking downstream.

    Calibration:
        Classification models often output probabilities that are poorly calibrated
        (e.g., a predicted 0.8 probability may only correspond to ~0.6 actual event
        frequency). Calibration adjusts these predicted probabilities so they better
        reflect true outcome frequencies.

        This implementation uses Platt scaling via
        `CalibratedClassifierCV(method="sigmoid", cv=3)`.

        Internally:
            1) Split (X_fit, y_fit) into 3 folds
            2) Train the base model on 2 folds
            3) Predict probabilities on the held-out fold
            4) Collect out-of-fold predictions
            5) Fit a sigmoid mapping that adjusts the model's pre-calibrated predicted 
                probabilities to calibrated probabilities

    Temporal caveat:
        The 3-fold CV used inside `CalibratedClassifierCV` ignores temporal ordering
        and may mix rows from different snapshot timestamps within (X_fit, y_fit).

        The leakage impact is limited because it occurs only within the training partition
        and does not affect downstream validation or test evaluation:
        a) Calibration only adjusts the model's pre-calibrated predicted probabilities.
        b) The temporal mixing arises from the CV folds used to generate
           out-of-fold predictions.
        c) True model evaluation (validation/test) still respects the
           temporal split and therefore remains leakage-free.
    """
    model = _build_propensity_model(model_name)
    calibrated_model = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
    calibrated_model.fit(X_fit, y_fit)
    propensity_scores = calibrated_model.predict_proba(X_pred)[:, 1]
    return calibrated_model, propensity_scores


def _fit_validation_propensity_model_wrapper(
    train_matrix,
    purchase_train_labels: np.ndarray,
    validation_matrix,
    purchase_validation_labels: np.ndarray,
    model_name: str,
    bins: int = 10,
) -> dict:
    """What: Fit one propensity model on train and score validation rows.
    Why: Produces validation metrics for frozen model selection.
    """
    calibrated_model, propensity_scores = _fit_and_score_propensity_model(
        train_matrix,
        purchase_train_labels,
        validation_matrix,
        model_name=model_name,
    )
    candidate = {
        "model_name": model_name,
        "model": calibrated_model,
        "propensity_scores": propensity_scores,
        "roc_auc": float(roc_auc_score(purchase_validation_labels, propensity_scores)),
        "average_precision": float(average_precision_score(purchase_validation_labels, propensity_scores)),
    }
    quality_metrics = _propensity_quality_metrics(np.asarray(purchase_validation_labels, dtype=int), candidate["propensity_scores"], bins=bins)
    return {**candidate, **quality_metrics}


def _fit_test_propensity_model_wrapper(
    train_val_matrix,
    purchase_train_val_labels: np.ndarray,
    test_matrix,
    selected_model_name: str,
):
    """What: Refit the selected propensity model on train+validation and score test.
    Why: Runs test-stage scoring after validation has frozen model choice.
    """
    selected_model, test_propensity_scores = _fit_and_score_propensity_model(
        train_val_matrix,
        purchase_train_val_labels,
        test_matrix,
        model_name=selected_model_name,
    )
    return selected_model, test_propensity_scores


def _fit_conditional_revenue_model(
    X_fit,
    y_purchase: np.ndarray,
    y_revenue: np.ndarray,
    X_pred,
    requested_revenue_model_name: str = "xgboost_regressor_conditional_revenue",
) -> tuple[object | None, str, float, np.ndarray]:
    """What: Fit conditional revenue E(revenue | purchase=1, x) and score prediction rows.
    Why: Supplies the revenue component for expected-value ranking with safe fallback.

    Returns:
    - revenue_model: fitted XGBoost regressor, or None when constant fallback is used.
    - effective_revenue_model_name: actual model path used for predictions.
    - fallback_value: mean positive revenue from fit rows (used for constant fallback).
    - predictions: non-negative conditional revenue predictions for score rows.
    """
    # A) Derive positive-row training subset and fallback value.
    positive_indices = np.where(y_purchase == 1)[0]
    if positive_indices.size == 0: raise ValueError("No positive purchase rows for conditional revenue training.")
    positive_targets = y_revenue[positive_indices]
    fallback_value = float(positive_targets.mean())

    effective_revenue_model_name = requested_revenue_model_name
    revenue_model = None

    # B) Resolve requested path and apply fallback when XGB cannot be fit.
    if requested_revenue_model_name == "xgboost_regressor_conditional_revenue":
        if positive_indices.size >= 2:
            revenue_model = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            )
            revenue_model.fit(X_fit[positive_indices], positive_targets)
        else:
            effective_revenue_model_name = "constant_mean_positive_revenue"

    # C) Generate predictions from the effective path and return outputs.
    if effective_revenue_model_name == "xgboost_regressor_conditional_revenue":
        predictions = np.maximum(0.0, revenue_model.predict(X_pred)).astype(float, copy=False)
    else:
        predictions = np.full(X_pred.shape[0], max(0.0, fallback_value), dtype=float)
    return revenue_model, effective_revenue_model_name, fallback_value, predictions


def _fit_validation_conditional_revenue_model_wrapper(
    train_matrix,
    train_purchase_labels: np.ndarray,
    train_conditional_revenue_labels: np.ndarray,
    validation_matrix,
    validation_purchase_labels: np.ndarray,
    validation_conditional_revenue_labels: np.ndarray,
) -> tuple[object | None, str, float, np.ndarray, dict]:
    """What: Fit/score conditional revenue for validation rows (request XGB, allow fallback).
    Why: Freezes effective revenue model choice from validation stage.
    """
    revenue_model, effective_revenue_model_name, fallback_value, predictions = _fit_conditional_revenue_model(
        train_matrix,
        train_purchase_labels,
        train_conditional_revenue_labels,
        validation_matrix,
        requested_revenue_model_name="xgboost_regressor_conditional_revenue",
    )
    positive_indices = np.where(validation_purchase_labels == 1)[0]
    revenue_quality = {"row_count": 0, "rmse": None, "mae": None, "mape": None}
    if positive_indices.size > 0:
        actuals = validation_conditional_revenue_labels[positive_indices]
        selected_predictions = predictions[positive_indices]
        mape = _safe_mape(actuals, selected_predictions)
        revenue_quality = {
            "row_count": int(positive_indices.size),
            "rmse": round(float(np.sqrt(mean_squared_error(actuals, selected_predictions))), 6),
            "mae": round(float(mean_absolute_error(actuals, selected_predictions)), 6),
            "mape": round(mape, 6) if mape is not None else None,
        }
    return revenue_model, effective_revenue_model_name, fallback_value, predictions, revenue_quality


def _fit_test_conditional_revenue_model_wrapper(
    train_val_matrix,
    train_val_purchase_labels: np.ndarray,
    train_val_conditional_revenue_labels: np.ndarray,
    test_matrix,
    test_purchase_labels: np.ndarray,
    test_conditional_revenue_labels: np.ndarray,
    frozen_revenue_model_name: str,
) -> tuple[object | None, str, float, np.ndarray, dict]:
    """What: Fit/score conditional revenue for test rows using frozen validation choice.
    Why: Ensures no model-type reselection during test stage.
    """
    revenue_model, effective_revenue_model_name, fallback_value, predictions = _fit_conditional_revenue_model(
        train_val_matrix,
        train_val_purchase_labels,
        train_val_conditional_revenue_labels,
        test_matrix,
        requested_revenue_model_name=frozen_revenue_model_name,
    )
    positive_indices = np.where(test_purchase_labels == 1)[0]
    revenue_quality = {"row_count": 0, "rmse": None, "mae": None, "mape": None}
    if positive_indices.size > 0:
        actuals = test_conditional_revenue_labels[positive_indices]
        selected_predictions = predictions[positive_indices]
        mape = _safe_mape(actuals, selected_predictions)
        revenue_quality = {
            "row_count": int(positive_indices.size),
            "rmse": round(float(np.sqrt(mean_squared_error(actuals, selected_predictions))), 6),
            "mae": round(float(mean_absolute_error(actuals, selected_predictions)), 6),
            "mape": round(mape, 6) if mape is not None else None,
        }
    return revenue_model, effective_revenue_model_name, fallback_value, predictions, revenue_quality
