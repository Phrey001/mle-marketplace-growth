"""Shared constants for purchase propensity training and evaluation scripts."""

# Window contracts used across feature-store and ML layers.
ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}  # What: supported future label horizons. Why: config and training should reject unsupported window choices.
ALLOWED_FEATURE_LOOKBACK_WINDOWS = {60, 90, 120}  # What: supported feature lookback horizons. Why: keeps feature profiles aligned to the implemented gold schema.

# Training-time preprocessing contract.
SPEND_CAP_QUANTILE = 0.99  # What: quantile used to cap spend features. Why: reduces outlier skew in training and serving inputs.

# Fixed structural sweep space for window sensitivity.
SENSITIVITY_MODEL_NAMES = ("logistic_regression", "xgboost")  # What: propensity models compared during sensitivity. Why: sweep should use one fixed comparable model set.
DEFAULT_LOOKBACK_FOR_WINDOW_SWEEP = 90  # What: fixed lookback used in prediction-window sensitivity. Why: isolates label-window effects before lookback comparison.
