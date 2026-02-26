from __future__ import annotations


def _expected_calibration_error(labels: list[int], scores: list[float], bins: int = 10) -> float:
    total_rows = len(labels)
    weighted_gap = 0.0
    for idx in range(bins):
        lower, upper = idx / bins, (idx + 1) / bins
        in_bin = [row_idx for row_idx, value in enumerate(scores) if (lower <= value <= upper if idx == bins - 1 else lower <= value < upper)]
        if not in_bin: continue
        avg_score = sum(scores[row_idx] for row_idx in in_bin) / len(in_bin)
        avg_label = sum(labels[row_idx] for row_idx in in_bin) / len(in_bin)
        weighted_gap += abs(avg_score - avg_label) * (len(in_bin) / total_rows)
    return float(weighted_gap)


def _safe_mape(actuals: list[float], predictions: list[float]) -> float | None:
    non_zero_pairs = [(actual, prediction) for actual, prediction in zip(actuals, predictions, strict=True) if actual != 0]
    if not non_zero_pairs: return None
    return float(sum(abs((actual - prediction) / actual) for actual, prediction in non_zero_pairs) / len(non_zero_pairs))
