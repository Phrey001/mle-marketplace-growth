from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb


def _quantile(values: list[float], q: float) -> float:
    """Compute a linear-interpolated quantile for a list of floats.
    Used by: train.py, window_sensitivity.py.
    """
    sorted_values = sorted(values)
    if not sorted_values: raise ValueError("Cannot compute quantile on empty values.")
    if q <= 0: return sorted_values[0]
    if q >= 1: return sorted_values[-1]
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _apply_spend_cap(rows: list[dict], spend_feature: str, spend_cap_value: float) -> None:
    """Clamp a spend feature to the cap value in-place.
    Used by: train.py.
    """
    for row in rows:
        row["features"][spend_feature] = min(row["features"][spend_feature], spend_cap_value)


def _stable_ratio(key: str) -> float:
    """Map a string key to a stable pseudo-random ratio in [0, 1).
    Used by: _policy_scores (train.py).
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _load_snapshot_rows(
    input_paths: Path | list[Path],
    feature_columns: list[str],
    purchase_label_column: str,
    revenue_label_column: str,
) -> list[dict]:
    """Load snapshot parquet rows and assemble feature/label dicts (pre-split).
    Used by: train.py.
    """
    paths = [input_paths] if isinstance(input_paths, Path) else input_paths
    if not paths: raise ValueError("At least one training dataset path is required.")
    rows = []
    for input_path in paths:
        # 1) Load parquet rows via DuckDB.
        connection = duckdb.connect(database=":memory:")
        try:
            cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(input_path)])
            columns = [col[0] for col in cursor.description]
            source_rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        finally:
            connection.close()
        # 2) Shape rows into model-ready dicts (features + labels).
        for row in source_rows:
            rows.append(
                {
                    "user_id": str(row["user_id"]),
                    "as_of_date": str(row["as_of_date"]),
                    "features": {feature: float(row[feature]) for feature in feature_columns} | {"country": str(row["country"])},
                    "purchase_label": float(row[purchase_label_column]),
                    "revenue_label": float(row[revenue_label_column]),
                }
            )
    if not rows:
        joined_paths = ", ".join(str(path) for path in paths)
        raise ValueError(f"No rows found in input dataset(s): {joined_paths}")
    return rows


def _split_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], str]:
    """Split rows into strict 10/1/1 train/validation/test by as_of_date.
    Used by: train.py.
    """
    unique_dates = sorted({row["as_of_date"] for row in rows})
    if len(unique_dates) != 12: raise ValueError("Strict split requires exactly 12 unique as_of_date snapshots " f"(got {len(unique_dates)}).")
    train_dates, validation_dates, test_dates = set(unique_dates[:10]), {unique_dates[10]}, {unique_dates[11]}
    split_desc = (
        f"out_of_time_10_1_1_train_dates={sorted(train_dates)};"
        f"validation_dates={sorted(validation_dates)};"
        f"test_dates={sorted(test_dates)}"
    )
    return (
        [row for row in rows if row["as_of_date"] in train_dates],
        [row for row in rows if row["as_of_date"] in validation_dates],
        [row for row in rows if row["as_of_date"] in test_dates],
        split_desc,
    )


def _policy_scores(
    rows: list[dict],
    propensity_scores: list[float],
    predicted_conditional_revenue: list[float],
    feature_lookback_days: int,
) -> tuple[list[float], list[float], list[float]]:
    """Compute expected-value, random, and RFM policy scores.
    Used by: train.py.
    """
    expected_value_scores = [float(score) * float(revenue) for score, revenue in zip(propensity_scores, predicted_conditional_revenue, strict=True)]
    random_scores = [1.0 - _stable_ratio(f'{row["user_id"]}|{row["as_of_date"]}|policy_random') for row in rows]
    freq_feature, monetary_feature = f"frequency_{feature_lookback_days}d", f"monetary_{feature_lookback_days}d"
    rfm_scores = [
        (1.0 / (1.0 + row["features"]["recency_days"])) + 0.5 * row["features"][freq_feature] + 0.01 * row["features"][monetary_feature]
        for row in rows
    ]
    return expected_value_scores, random_scores, rfm_scores
