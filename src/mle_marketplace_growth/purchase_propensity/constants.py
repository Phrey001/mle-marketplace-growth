"""Shared constants for purchase propensity training and evaluation scripts."""

# Window contracts used across feature-store and ML layers.
ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}
ALLOWED_FEATURE_LOOKBACK_WINDOWS = {60, 90, 120}

# Training-time preprocessing contract.
SPEND_CAP_QUANTILE = 0.99

# Fixed structural sweep space for window sensitivity.
SENSITIVITY_PREDICTION_WINDOWS = (30, 60, 90)
SENSITIVITY_LOOKBACK_WINDOWS = (60, 90, 120)
SENSITIVITY_MODEL_NAMES = ("logistic_regression", "xgboost")
DEFAULT_LOOKBACK_FOR_WINDOW_SWEEP = 90
