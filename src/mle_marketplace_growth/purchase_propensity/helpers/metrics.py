from __future__ import annotations

"""Metric helpers for purchase propensity evaluation."""

import numpy as np
from sklearn.metrics import brier_score_loss


def _expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    """Compute expected calibration error (ECE) over probability scores.

    What: ECE answers "when the model says 20% chance, does it happen about 20% of the time?"
    Inputs: labels are 0/1 outcomes; scores are predicted probabilities in [0, 1].
    Output: a single scalar score (this helper does not return per-bin gaps).
    """
    labels_arr = np.asarray(labels, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    # Assign each score to a bin index (0..bins-1); clip handles score==1.0.
    # np.digitize maps each value to the index of the bin edge it falls into.
    # right=False means a value exactly equal to a bin edge goes into the lower bin.
    # The score==1.0 case would otherwise map to bin index 10 (outside 0–9), so we clip it into the final bin (0.9–1.0 for bins=10).
    # digitize returns indices in 1..bins, so we subtract 1 for 0-based bin ids.
    # Example: scores [0.02, 0.23, 0.58, 0.99, 1.0] with bins=10 -> bin_ids [0,2,5,9,9].
    bin_ids = np.clip(np.digitize(scores_arr, bin_edges, right=False) - 1, 0, bins - 1)

    weighted_gap = 0.0
    for bin_idx in range(bins):
        in_bin = bin_ids == bin_idx
        if not in_bin.any():
            # No scores landed in this bin, so skip to avoid empty-mean warnings.
            continue
        # Compare average predicted probability vs actual label rate within the bin.
        # Per-bin gap is |avg_score - avg_label|; bin weight is in_bin.mean() (fraction of rows in the bin).
        avg_score = float(scores_arr[in_bin].mean())
        avg_label = float(labels_arr[in_bin].mean())
        # Weight the gap by how many rows fall into the bin.
        weighted_gap += abs(avg_score - avg_label) * float(in_bin.mean())
    return float(weighted_gap)


def _safe_mape(actuals: np.ndarray, predictions: np.ndarray) -> float | None:
    """Compute MAPE while ignoring rows with zero actuals."""
    actuals_arr = np.asarray(actuals, dtype=float)
    predictions_arr = np.asarray(predictions, dtype=float)
    non_zero_mask = actuals_arr != 0
    if not np.any(non_zero_mask): return None
    return float(np.mean(np.abs((actuals_arr[non_zero_mask] - predictions_arr[non_zero_mask]) / actuals_arr[non_zero_mask])))


def _propensity_quality_metrics(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Compute shared propensity quality metrics from labels and probability scores."""
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    top_decile_count = max(1, int(0.1 * labels_arr.shape[0]))
    top_decile_indices = np.argsort(scores_arr)[-top_decile_count:]
    base_positive_rate = float(labels_arr.mean())
    top_decile_positive_rate = float(labels_arr[top_decile_indices].mean())
    top_decile_lift = top_decile_positive_rate / base_positive_rate if base_positive_rate > 0 else 0.0
    return {
        "base_positive_rate": base_positive_rate,
        "top_decile_positive_rate": top_decile_positive_rate,
        "top_decile_lift": float(top_decile_lift),
        "brier_score": float(brier_score_loss(labels_arr, scores_arr)),
        "ece_10_bin": _expected_calibration_error(labels_arr, scores_arr, bins=bins),
    }
