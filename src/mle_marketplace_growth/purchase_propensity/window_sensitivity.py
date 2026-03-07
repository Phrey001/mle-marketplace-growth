"""Run 30/60/90-day label-window sensitivity for propensity modeling."""

import argparse
import calendar
import json
from bisect import bisect_right
from datetime import date, timedelta
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from mle_marketplace_growth.purchase_propensity.helpers.data import _quantile
from mle_marketplace_growth.purchase_propensity.helpers.metrics import _expected_calibration_error
from mle_marketplace_growth.purchase_propensity.helpers.modeling import _build_model

FEATURE_KEYS_BY_LOOKBACK = {
    60: ["recency_days", "frequency_30d", "frequency_60d", "monetary_30d", "monetary_60d", "avg_basket_value_60d", "country"],
    90: ["recency_days", "frequency_30d", "frequency_90d", "monetary_30d", "monetary_90d", "avg_basket_value_90d", "country"],
    120: ["recency_days", "frequency_30d", "frequency_120d", "monetary_30d", "monetary_120d", "avg_basket_value_120d", "country"],
}
ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}
FIXED_WINDOWS = [30, 60, 90]
SPEND_CAP_QUANTILE = 0.99
CALIBRATION_METHOD = "sigmoid"


# ===== Shared Utilities =====
def _add_month(current: date) -> date:
    year = current.year + (1 if current.month == 12 else 0)
    month = 1 if current.month == 12 else current.month + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _shift_month(current: date, delta_months: int) -> date:
    month_index = current.month - 1 + delta_months
    year = current.year + month_index // 12
    month = month_index % 12 + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _panel_paths(panel_root: Path, panel_end_date: date) -> list[Path]:
    start_date = _shift_month(panel_end_date, -11)
    paths: list[Path] = []
    current = start_date
    for _ in range(12):
        paths.append(panel_root / f"as_of_date={current.isoformat()}" / "propensity_train_dataset.parquet")
        current = _add_month(current)
    return paths


def _strict_split_indices(rows: list[dict]) -> tuple[list[int], list[int], list[int], str]:
    unique_dates = sorted({row["as_of_date"] for row in rows})
    if len(unique_dates) != 12: raise ValueError("Strict split requires exactly 12 unique as_of_date snapshots " f"(got {len(unique_dates)}).")
    train_dates = set(unique_dates[:10])
    validation_dates = {unique_dates[10]}
    test_dates = {unique_dates[11]}
    train_indices = [idx for idx, row in enumerate(rows) if row["as_of_date"] in train_dates]
    validation_indices = [idx for idx, row in enumerate(rows) if row["as_of_date"] in validation_dates]
    test_indices = [idx for idx, row in enumerate(rows) if row["as_of_date"] in test_dates]
    split_description = (
        f"out_of_time_10_1_1_train_dates={sorted(train_dates)};"
        f"validation_dates={sorted(validation_dates)};"
        f"test_dates={sorted(test_dates)}"
    )
    return train_indices, validation_indices, test_indices, split_description


# ===== Data Loading =====
def _read_feature_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    connection = duckdb.connect(database=":memory:")
    try:
        for path in paths:
            cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
            columns = [col[0] for col in cursor.description]
            rows.extend(dict(zip(columns, row, strict=True)) for row in cursor.fetchall())
    finally:
        connection.close()
    return rows


def _read_event_rows(path: Path) -> list[dict]:
    connection = duckdb.connect(database=":memory:")
    try:
        cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()


def _load_positive_event_dates(path: Path) -> dict[str, list[date]]:
    rows = _read_event_rows(path)
    events_by_user: dict[str, list[date]] = {}
    for row in rows:
        user_id = str(row["user_id"])
        if not user_id:
            continue
        if float(row["quantity"]) <= 0:
            continue
        event_day = date.fromisoformat(str(row["event_date"]))
        events_by_user.setdefault(user_id, []).append(event_day)
    for user_id in events_by_user:
        events_by_user[user_id].sort()
    return events_by_user


def _load_feature_rows(rows: list[dict]) -> list[dict]:
    if not rows: raise ValueError("No rows in feature dataset.")
    required_columns = {
        "user_id",
        "as_of_date",
        "country",
        "recency_days",
        "frequency_30d",
        "frequency_60d",
        "frequency_90d",
        "frequency_120d",
        "monetary_30d",
        "monetary_60d",
        "monetary_90d",
        "monetary_120d",
        "avg_basket_value_60d",
        "avg_basket_value_90d",
        "avg_basket_value_120d",
    }
    missing = sorted(required_columns - set(rows[0].keys()))
    if missing: raise ValueError(f"Missing required feature columns for strict window sensitivity: {missing}")
    return [
        {
            "user_id": str(row["user_id"]),
            "as_of_date": str(row["as_of_date"]),
            "as_of_date_obj": date.fromisoformat(str(row["as_of_date"])),
            "features": {
                "recency_days": float(row["recency_days"]),
                "frequency_30d": float(row["frequency_30d"]),
                "frequency_60d": float(row["frequency_60d"]),
                "frequency_90d": float(row["frequency_90d"]),
                "frequency_120d": float(row["frequency_120d"]),
                "monetary_30d": float(row["monetary_30d"]),
                "monetary_60d": float(row["monetary_60d"]),
                "monetary_90d": float(row["monetary_90d"]),
                "monetary_120d": float(row["monetary_120d"]),
                "avg_basket_value_60d": float(row["avg_basket_value_60d"]),
                "avg_basket_value_90d": float(row["avg_basket_value_90d"]),
                "avg_basket_value_120d": float(row["avg_basket_value_120d"]),
                "country": row["country"],
            },
        }
        for row in rows
    ]


# ===== Label + Metric Utilities =====
def _has_purchase_in_window(events: list[date], as_of_day: date, window_days: int) -> int:
    start_day = as_of_day + timedelta(days=1)
    end_day = as_of_day + timedelta(days=window_days)
    first_idx = bisect_right(events, as_of_day)
    return 1 if first_idx < len(events) and events[first_idx] <= end_day and events[first_idx] >= start_day else 0


def _inter_purchase_gap_days(events_by_user: dict[str, list[date]]) -> list[int]:
    gaps = []
    for events in events_by_user.values():
        if len(events) < 2: continue
        for left, right in zip(events[:-1], events[1:], strict=True):
            gap_days = (right - left).days
            if gap_days > 0: gaps.append(gap_days)
    return gaps


# ===== Model Evaluation =====
# Model evaluation is intentionally split into three steps for readability:
# 1) build capped train/validation matrices, 2) short-circuit single-class splits,
# 3) evaluate candidate models and return comparable metrics.
def _build_split_matrices(
    rows: list[dict],
    labels: list[int],
    train_indices: list[int],
    validation_indices: list[int],
    feature_keys: list[str],
    spend_cap_quantile: float,
) -> tuple[list[int], list[int], object, object, float]:
    train_labels = [labels[idx] for idx in train_indices]
    validation_labels = [labels[idx] for idx in validation_indices]

    vectorizer = DictVectorizer(sparse=True)
    train_features = [{key: rows[idx]["features"][key] for key in feature_keys} for idx in train_indices]
    validation_features = [{key: rows[idx]["features"][key] for key in feature_keys} for idx in validation_indices]
    spend_feature_key = [key for key in feature_keys if key.startswith("monetary_")][-1]
    spend_cap_value = _quantile([float(features[spend_feature_key]) for features in train_features], spend_cap_quantile)
    train_features = [{**features, spend_feature_key: min(float(features[spend_feature_key]), spend_cap_value)} for features in train_features]
    validation_features = [{**features, spend_feature_key: min(float(features[spend_feature_key]), spend_cap_value)} for features in validation_features]
    return (
        train_labels,
        validation_labels,
        vectorizer.fit_transform(train_features),
        vectorizer.transform(validation_features),
        float(spend_cap_value),
    )


def _single_class_results(
    train_labels: list[int],
    validation_labels: list[int],
    train_rows: int,
    validation_rows: int,
    spend_cap_value: float,
    calibration_method: str,
    feature_keys: list[str],
) -> list[dict]:
    reason = "single_class_in_strict_split"
    train_rate = round(sum(train_labels) / len(train_labels), 6)
    validation_rate = round(sum(validation_labels) / len(validation_labels), 6)
    shared = {
        "roc_auc": 0.0,
        "average_precision": 0.0,
        "top_decile_lift": 0.0,
        "brier_score": 1.0,
        "ece_10_bin": 1.0,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "positive_rate_train": train_rate,
        "positive_rate_validation": validation_rate,
        "spend_cap_value": round(float(spend_cap_value), 6),
        "calibration_method": calibration_method,
        "feature_keys_used": feature_keys,
        "status": reason,
    }
    return [
        {
            "model_name": model_name,
            **shared,
        }
        for model_name in ["logistic_regression", "xgboost"]
    ]


def _evaluate_candidate_models(
    train_matrix,
    validation_matrix,
    train_labels: list[int],
    validation_labels: list[int],
    train_rows: int,
    validation_rows: int,
    spend_cap_value: float,
    calibration_method: str,
    feature_keys: list[str],
) -> list[dict]:
    results = []
    train_rate = round(sum(train_labels) / len(train_labels), 6)
    validation_rate = round(sum(validation_labels) / len(validation_labels), 6)
    base_positive_rate = sum(validation_labels) / len(validation_labels)
    top_decile_count = max(1, int(0.1 * len(validation_labels)))
    shared = {
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "positive_rate_train": train_rate,
        "positive_rate_validation": validation_rate,
        "spend_cap_value": round(float(spend_cap_value), 6),
        "calibration_method": calibration_method,
        "feature_keys_used": feature_keys,
    }
    for model_name in ["logistic_regression", "xgboost"]:
        model = _build_model(model_name)
        calibrated_model = CalibratedClassifierCV(estimator=model, method=calibration_method, cv=3) if calibration_method != "none" else model
        calibrated_model.fit(train_matrix, train_labels)
        scores = calibrated_model.predict_proba(validation_matrix)[:, 1]
        top_decile_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_decile_count]
        top_decile_positive_rate = sum(validation_labels[idx] for idx in top_decile_indices) / top_decile_count
        results.append(
            {
                "model_name": model_name,
                "roc_auc": round(float(roc_auc_score(validation_labels, scores)), 6),
                "average_precision": round(float(average_precision_score(validation_labels, scores)), 6),
                "top_decile_lift": round(float(top_decile_positive_rate / base_positive_rate if base_positive_rate > 0 else 0.0), 6),
                "brier_score": round(float(brier_score_loss(validation_labels, scores)), 6),
                "ece_10_bin": round(_expected_calibration_error(validation_labels, scores.tolist(), bins=10), 6),
                **shared,
            }
        )
    return results


def _evaluate_models(
    rows: list[dict],
    labels: list[int],
    train_indices: list[int],
    validation_indices: list[int],
    spend_cap_quantile: float,
    calibration_method: str,
    feature_keys: list[str],
) -> tuple[str, list[dict]]:
    if not train_indices or not validation_indices: raise ValueError("Empty train/validation split for strict 10/1/1 chronology.")
    train_labels, validation_labels, train_matrix, validation_matrix, spend_cap_value = _build_split_matrices(
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        feature_keys=feature_keys,
        spend_cap_quantile=spend_cap_quantile,
    )
    if len(set(train_labels)) < 2 or len(set(validation_labels)) < 2:
        return "unavailable", _single_class_results(
            train_labels=train_labels,
            validation_labels=validation_labels,
            train_rows=len(train_indices),
            validation_rows=len(validation_indices),
            spend_cap_value=spend_cap_value,
            calibration_method=calibration_method,
            feature_keys=feature_keys,
        )
    results = _evaluate_candidate_models(
        train_matrix=train_matrix,
        validation_matrix=validation_matrix,
        train_labels=train_labels,
        validation_labels=validation_labels,
        train_rows=len(train_indices),
        validation_rows=len(validation_indices),
        spend_cap_value=spend_cap_value,
        calibration_method=calibration_method,
        feature_keys=feature_keys,
    )
    best_model = max(results, key=lambda row: row["average_precision"])["model_name"]
    return best_model, results


# ===== Visualization =====
def _write_validation_dashboard(output: dict, output_path: Path) -> None:
    prediction_rows = output.get("window_sensitivity", [])
    lookback_rows = output.get("feature_window_validation", [])
    if not prediction_rows or not lookback_rows: return

    window_axis = [row["window_days"] for row in prediction_rows]
    best_window_metrics = [
        max(row["model_results"], key=lambda item: item["average_precision"]) for row in prediction_rows
    ]
    lookback_axis = [row["feature_lookback_days"] for row in lookback_rows]
    best_lookback_metrics = [
        max(row["model_results"], key=lambda item: item["average_precision"]) for row in lookback_rows
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metric_specs = [
        ("average_precision", "PR-AUC"),
        ("top_decile_lift", "Top-Decile Lift"),
        ("brier_score", "Brier Score"),
        ("ece_10_bin", "ECE (10-bin)"),
    ]
    for axis, (metric_key, title) in zip(axes.flat, metric_specs, strict=True):
        axis.plot(window_axis, [row[metric_key] for row in best_window_metrics], marker="o")
        axis.plot(lookback_axis, [row[metric_key] for row in best_lookback_metrics], marker="o")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("Window days")
    axes[0, 0].legend(["Prediction window", "Feature lookback"], loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote window validation dashboard: {output_path}")


# ===== Entry Point =====
def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Run 30/60/90-day sensitivity for propensity models.")
    parser.add_argument(
        "--input-path",
        action="append",
        default=None,
        help="Path to strict training-panel parquet (repeat --input-path for multi-snapshot panel)",
    )
    parser.add_argument(
        "--panel-root",
        default=None,
        help="Root folder of panel snapshots (e.g. data/.../propensity_train_dataset); auto-loads strict 12 months with --panel-end-date",
    )
    parser.add_argument("--panel-end-date", default=None, help="Panel end anchor (YYYY-MM-DD) used with --panel-root")
    parser.add_argument("--events-path", default="data/silver/transactions_line_items/transactions_line_items.parquet", help="Path to silver transactions parquet (for recalculating labels by window)")
    parser.add_argument("--output-json", default="artifacts/purchase_propensity/window_sensitivity.json", help="Path to sensitivity summary JSON")
    parser.add_argument("--output-plot", default="artifacts/purchase_propensity/window_validation_dashboard.png", help="Path to window validation dashboard PNG")
    args = parser.parse_args()

    # ===== Input Checks =====
    if args.input_path:
        feature_paths = [Path(path) for path in args.input_path]
    elif args.panel_root and args.panel_end_date:
        feature_paths = _panel_paths(Path(args.panel_root), date.fromisoformat(args.panel_end_date))
    else:
        feature_paths = [
            Path("data/gold/feature_store/purchase_propensity/propensity_train_dataset/as_of_date=2011-11-09/propensity_train_dataset.parquet")
        ]
    events_path = Path(args.events_path)
    missing_features = [path for path in feature_paths if not path.exists()]
    if missing_features: raise FileNotFoundError(f"Input path not found: {missing_features[0]}")
    if not events_path.exists(): raise FileNotFoundError(f"Events path not found: {events_path}")

    windows = FIXED_WINDOWS
    if set(windows) != ALLOWED_PREDICTION_WINDOWS:
        raise ValueError("--windows must be exactly 30,60,90 for strict architecture alignment.")

    # ===== Load Data =====
    raw_rows = _read_feature_rows(feature_paths)
    if not raw_rows: raise ValueError(f"No rows in feature dataset(s): {', '.join(str(path) for path in feature_paths)}")
    rows = _load_feature_rows(raw_rows)

    # ===== Split + Label Schema Checks =====
    train_indices, validation_indices, _, split_description = _strict_split_indices(rows)
    events_by_user = _load_positive_event_dates(events_path)
    inter_purchase_gap_days = _inter_purchase_gap_days(events_by_user)
    required_label_columns = [f"label_purchase_{window_days}d" for window_days in windows]
    missing_labels = [column for column in required_label_columns if column not in raw_rows[0]]
    if missing_labels: raise ValueError(f"Missing required label columns for strict window sensitivity: {missing_labels}")

    # ===== Prediction-Window Model Evaluation =====
    labels_by_window = {}
    sensitivity_rows = []
    for window_days in windows:
        label_col = f"label_purchase_{window_days}d"
        labels = [int(float(row[label_col])) for row in raw_rows]
        if sum(labels) == 0: raise ValueError(f"No positive labels for window={window_days}.")
        labels_by_window[window_days] = labels

        best_model, model_results = _evaluate_models(
            rows,
            labels,
            train_indices,
            validation_indices,
            SPEND_CAP_QUANTILE,
            CALIBRATION_METHOD,
            FEATURE_KEYS_BY_LOOKBACK[90],
        )
        sensitivity_rows.append(
            {
                "window_days": window_days,
                "best_model_by_average_precision": best_model,
                "model_results": model_results,
            }
        )

    # ===== Prediction-Window Coverage Diagnostics =====
    prediction_window_validation = []
    if inter_purchase_gap_days:
        for window_days in windows:
            coverage = sum(1 for gap in inter_purchase_gap_days if gap <= window_days) / len(inter_purchase_gap_days)
            prediction_window_validation.append(
                {
                    "window_days": window_days,
                    "inter_purchase_gap_coverage": round(float(coverage), 6),
                }
            )

    # ===== Feature-Lookback Model Evaluation =====
    feature_window_validation = []
    prediction_window_for_feature_eval = 30 if 30 in labels_by_window else windows[0]
    for lookback_days in [60, 90, 120]:
        best_model, model_results = _evaluate_models(
            rows,
            labels_by_window[prediction_window_for_feature_eval],
            train_indices,
            validation_indices,
            SPEND_CAP_QUANTILE,
            CALIBRATION_METHOD,
            FEATURE_KEYS_BY_LOOKBACK[lookback_days],
        )
        feature_window_validation.append(
            {
                "feature_lookback_days": lookback_days,
                "prediction_window_days_for_eval": prediction_window_for_feature_eval,
                "feature_profile_note": "direct feature schema",
                "best_model_by_average_precision": best_model,
                "model_results": model_results,
            }
        )

    # ===== Freeze Decision + Output Write =====
    best_prediction_window = max(
        sensitivity_rows,
        key=lambda row: max(item["average_precision"] for item in row.get("model_results", [{"average_precision": 0.0}])),
    )
    best_lookback_window = max(
        feature_window_validation,
        key=lambda row: max(item["average_precision"] for item in row.get("model_results", [{"average_precision": 0.0}])),
    )
    best_prediction_model = max(best_prediction_window["model_results"], key=lambda row: float(row.get("average_precision", 0.0)))
    best_lookback_model = max(best_lookback_window["model_results"], key=lambda row: float(row.get("average_precision", 0.0)))
    best_prediction_model_name, best_prediction_model_ap = str(best_prediction_model["model_name"]), float(best_prediction_model["average_precision"])
    best_lookback_model_name, best_lookback_model_ap = str(best_lookback_model["model_name"]), float(best_lookback_model["average_precision"])
    # Freeze structural decisions from validation so downstream train/predict/eval
    # uses one fixed configuration for the demo cycle.
    freeze_decision = {
        "selected_prediction_window_days": int(best_prediction_window["window_days"]),
        "selected_feature_lookback_days": int(best_lookback_window["feature_lookback_days"]),
        "selected_propensity_model_name": best_lookback_model_name,
        "selection_rule": "maximize_validation_average_precision",
        "selection_summary": {
            "prediction_window_model": best_prediction_model_name,
            "prediction_window_average_precision": round(best_prediction_model_ap, 6),
            "lookback_model": best_lookback_model_name,
            "lookback_average_precision": round(best_lookback_model_ap, 6),
        },
    }
    sorted_gaps = sorted(inter_purchase_gap_days)
    output = {
        "input_paths": [str(path) for path in feature_paths],
        "events_path": str(events_path),
        "split_strategy": split_description,
        "spend_cap_quantile": SPEND_CAP_QUANTILE,
        "calibration_method": CALIBRATION_METHOD,
        "note": "offline sensitivity only; not causal promotional incrementality evidence",
        "inter_purchase_distribution": {
            "gap_observation_count": len(inter_purchase_gap_days),
            "median_gap_days": sorted_gaps[len(sorted_gaps) // 2] if sorted_gaps else None,
            "mean_gap_days": round(sum(inter_purchase_gap_days) / len(inter_purchase_gap_days), 6)
            if inter_purchase_gap_days
            else None,
        },
        "prediction_window_validation": prediction_window_validation,
        "feature_window_validation": feature_window_validation,
        "window_sensitivity": sensitivity_rows,
        "freeze_decision": freeze_decision,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote sensitivity summary: {output_path}")
    _write_validation_dashboard(output, Path(args.output_plot))


if __name__ == "__main__":
    main()
