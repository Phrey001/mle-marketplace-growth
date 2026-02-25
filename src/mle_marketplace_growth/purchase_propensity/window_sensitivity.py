"""Run 30/60/90-day label-window sensitivity for propensity modeling."""

import argparse
import csv
import json
from bisect import bisect_right
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURE_KEYS_BY_LOOKBACK = {
    60: ["recency_days", "frequency_30d", "frequency_60d", "monetary_30d", "monetary_60d", "avg_basket_value_60d", "country"],
    90: ["recency_days", "frequency_30d", "frequency_90d", "monetary_30d", "monetary_90d", "avg_basket_value_90d", "country"],
    120: ["recency_days", "frequency_30d", "frequency_120d", "monetary_30d", "monetary_120d", "avg_basket_value_120d", "country"],
}


# ===== Shared Utilities =====
def _strict_split_indices(rows: list[dict]) -> tuple[list[int], list[int], list[int], str]:
    unique_dates = sorted({row["as_of_date"] for row in rows})
    if len(unique_dates) != 12:
        raise ValueError(
            "Strict split requires exactly 12 unique as_of_date snapshots "
            f"(got {len(unique_dates)})."
        )
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
def _load_feature_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows in feature dataset: {path}")
    return [
        {
            "user_id": row["user_id"],
            "as_of_date": row["as_of_date"],
            "as_of_date_obj": date.fromisoformat(row["as_of_date"]),
            "features": {
                "recency_days": float(row["recency_days"]),
                "frequency_30d": float(row["frequency_30d"]),
                "frequency_60d": float(row.get("frequency_60d") or row["frequency_90d"]),
                "frequency_90d": float(row["frequency_90d"]),
                "frequency_120d": float(row.get("frequency_120d") or row["frequency_90d"]),
                "monetary_30d": float(row["monetary_30d"]),
                "monetary_60d": float(row.get("monetary_60d") or row["monetary_90d"]),
                "monetary_90d": float(row["monetary_90d"]),
                "monetary_120d": float(row.get("monetary_120d") or row["monetary_90d"]),
                "avg_basket_value_60d": float(row.get("avg_basket_value_60d") or row["avg_basket_value_90d"]),
                "avg_basket_value_90d": float(row["avg_basket_value_90d"]),
                "avg_basket_value_120d": float(row.get("avg_basket_value_120d") or row["avg_basket_value_90d"]),
                "country": row["country"],
            },
        }
        for row in rows
    ]


def _load_positive_event_dates(path: Path) -> dict[str, list[date]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        events_by_user = {}
        for row in rows:
            user_id = row["user_id"]
            if not user_id:
                continue
            if float(row["quantity"]) <= 0:
                continue
            event_day = date.fromisoformat(row["event_date"])
            events_by_user.setdefault(user_id, []).append(event_day)
    for user_id in events_by_user:
        events_by_user[user_id].sort()
    return events_by_user


# ===== Label + Metric Utilities =====
def _has_purchase_in_window(events: list[date], as_of_day: date, window_days: int) -> int:
    start_day = as_of_day + timedelta(days=1)
    end_day = as_of_day + timedelta(days=window_days)
    first_idx = bisect_right(events, as_of_day)
    return 1 if first_idx < len(events) and events[first_idx] <= end_day and events[first_idx] >= start_day else 0


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


def _inter_purchase_gap_days(events_by_user: dict[str, list[date]]) -> list[int]:
    gaps = []
    for events in events_by_user.values():
        if len(events) < 2:
            continue
        for left, right in zip(events[:-1], events[1:], strict=True):
            gap_days = (right - left).days
            if gap_days > 0:
                gaps.append(gap_days)
    return gaps


# ===== Model Evaluation =====
def _evaluate_models(
    rows: list[dict],
    labels: list[int],
    train_indices: list[int],
    validation_indices: list[int],
    spend_cap_quantile: float,
    calibration_method: str,
    feature_keys: list[str],
) -> tuple[str, list[dict]]:
    if not train_indices or not validation_indices:
        raise ValueError("Empty train/validation split for strict 10/1/1 chronology.")

    train_labels = [labels[idx] for idx in train_indices]
    validation_labels = [labels[idx] for idx in validation_indices]

    vectorizer = DictVectorizer(sparse=True)
    train_features = [{key: rows[idx]["features"][key] for key in feature_keys} for idx in train_indices]
    validation_features = [{key: rows[idx]["features"][key] for key in feature_keys} for idx in validation_indices]
    spend_feature_key = [key for key in feature_keys if key.startswith("monetary_")][-1]
    spend_cap_value = _quantile([float(features[spend_feature_key]) for features in train_features], spend_cap_quantile)
    train_features = [
        {**features, spend_feature_key: min(float(features[spend_feature_key]), spend_cap_value)}
        for features in train_features
    ]
    validation_features = [
        {**features, spend_feature_key: min(float(features[spend_feature_key]), spend_cap_value)}
        for features in validation_features
    ]
    train_matrix = vectorizer.fit_transform(train_features)
    validation_matrix = vectorizer.transform(validation_features)

    if len(set(train_labels)) < 2 or len(set(validation_labels)) < 2:
        reason = "single_class_in_strict_split"
        unavailable = []
        for model_name in ["logistic_regression", "xgboost"]:
            unavailable.append(
                {
                    "model_name": model_name,
                    "roc_auc": 0.0,
                    "average_precision": 0.0,
                    "top_decile_lift": 0.0,
                    "brier_score": 1.0,
                    "ece_10_bin": 1.0,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "positive_rate_train": round(sum(train_labels) / len(train_labels), 6),
                    "positive_rate_validation": round(sum(validation_labels) / len(validation_labels), 6),
                    "spend_cap_value": round(float(spend_cap_value), 6),
                    "calibration_method": calibration_method,
                    "feature_keys_used": feature_keys,
                    "status": reason,
                }
            )
        return "unavailable", unavailable

    model_candidates = [
        (
            "logistic_regression",
            make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            ),
        ),
    ]

    results = []
    for model_name, model in model_candidates:
        calibrated_model = model
        if calibration_method != "none":
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                method=calibration_method,
                cv=3,
            )
        calibrated_model.fit(train_matrix, train_labels)
        scores = calibrated_model.predict_proba(validation_matrix)[:, 1]
        base_positive_rate = sum(validation_labels) / len(validation_labels)
        top_decile_count = max(1, int(0.1 * len(validation_labels)))
        top_decile_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_decile_count]
        top_decile_positive_rate = sum(validation_labels[idx] for idx in top_decile_indices) / top_decile_count
        top_decile_lift = top_decile_positive_rate / base_positive_rate if base_positive_rate > 0 else 0.0
        results.append(
            {
                "model_name": model_name,
                "roc_auc": round(float(roc_auc_score(validation_labels, scores)), 6),
                "average_precision": round(float(average_precision_score(validation_labels, scores)), 6),
                "top_decile_lift": round(float(top_decile_lift), 6),
                "brier_score": round(float(brier_score_loss(validation_labels, scores)), 6),
                "ece_10_bin": round(_expected_calibration_error(validation_labels, scores.tolist(), bins=10), 6),
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "positive_rate_train": round(sum(train_labels) / len(train_labels), 6),
                "positive_rate_validation": round(sum(validation_labels) / len(validation_labels), 6),
                "spend_cap_value": round(float(spend_cap_value), 6),
                "calibration_method": calibration_method,
                "feature_keys_used": feature_keys,
            }
        )
    best_model = max(results, key=lambda row: row["average_precision"])["model_name"]
    return best_model, results


# ===== Visualization =====
def _write_validation_dashboard(output: dict, output_path: Path) -> None:
    prediction_rows = output.get("window_sensitivity", [])
    lookback_rows = output.get("feature_window_validation", [])
    if not prediction_rows or not lookback_rows:
        return

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


def _best_model_by_ap(model_results: list[dict]) -> tuple[str, float]:
    best = max(model_results, key=lambda row: float(row.get("average_precision", 0.0)))
    return str(best["model_name"]), float(best["average_precision"])


# ===== Entry Point =====
def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Run 30/60/90-day sensitivity for propensity models.")
    parser.add_argument("--input-csv", default="artifacts/purchase_propensity/_tmp/propensity_train_dataset_merged.csv", help="Path to merged strict 12-snapshot training panel CSV")
    parser.add_argument("--events-csv", default="data/silver/transactions_line_items/transactions_line_items.csv", help="Path to silver transactions CSV (for recalculating labels by window)")
    parser.add_argument("--windows", default="30,60,90", help="Comma-separated label windows in days")
    parser.add_argument("--spend-cap-quantile", type=float, default=0.99, help="Quantile cap for monetary_90d to reduce extreme-spend dominance (0 < q <= 1).")
    parser.add_argument(
        "--calibration-method",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration method applied on train folds before validation scoring.",
    )
    parser.add_argument("--output-json", default="artifacts/purchase_propensity/window_sensitivity.json", help="Path to sensitivity summary JSON")
    parser.add_argument("--output-plot", default="artifacts/purchase_propensity/window_validation_dashboard.png", help="Path to window validation dashboard PNG")
    args = parser.parse_args()

    # ===== Input Checks =====
    if not 0.0 < args.spend_cap_quantile <= 1.0:
        raise ValueError("--spend-cap-quantile must be in (0, 1]")

    feature_path = Path(args.input_csv)
    events_path = Path(args.events_csv)
    if not feature_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {feature_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Events CSV not found: {events_path}")

    windows = sorted({int(value.strip()) for value in args.windows.split(",") if value.strip()})
    if not windows:
        raise ValueError("No windows provided.")

    # ===== Load Data =====
    rows = _load_feature_rows(feature_path)
    with feature_path.open("r", encoding="utf-8", newline="") as file:
        raw_rows = list(csv.DictReader(file))
    train_indices, validation_indices, _, split_description = _strict_split_indices(rows)
    events_by_user = _load_positive_event_dates(events_path)
    inter_purchase_gap_days = _inter_purchase_gap_days(events_by_user)

    # ===== Prediction-Window Evaluation =====
    labels_by_window = {}
    sensitivity_rows = []
    for window_days in windows:
        label_col = f"label_purchase_{window_days}d"
        labels = None
        if raw_rows and label_col in raw_rows[0]:
            labels = [int(float(row[label_col])) for row in raw_rows]
        if labels is None:
            labels = []
            for row in rows:
                events = events_by_user.get(row["user_id"], [])
                labels.append(_has_purchase_in_window(events, row["as_of_date_obj"], window_days))
        if sum(labels) == 0:
            raise ValueError(f"No positive labels for window={window_days}.")
        labels_by_window[window_days] = labels

        best_model, model_results = _evaluate_models(
            rows,
            labels,
            train_indices,
            validation_indices,
            args.spend_cap_quantile,
            args.calibration_method,
            FEATURE_KEYS_BY_LOOKBACK[90],
        )
        sensitivity_rows.append(
            {
                "window_days": window_days,
                "best_model_by_average_precision": best_model,
                "model_results": model_results,
            }
        )

    # ===== Window Coverage Check =====
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

    # ===== Feature-Lookback Evaluation =====
    feature_window_validation = []
    prediction_window_for_feature_eval = 30 if 30 in labels_by_window else windows[0]
    for lookback_days in [60, 90, 120]:
        best_model, model_results = _evaluate_models(
            rows,
            labels_by_window[prediction_window_for_feature_eval],
            train_indices,
            validation_indices,
            args.spend_cap_quantile,
            args.calibration_method,
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

    # ===== Write Outputs =====
    best_prediction_window = max(
        sensitivity_rows,
        key=lambda row: max(item["average_precision"] for item in row.get("model_results", [{"average_precision": 0.0}])),
    )
    best_lookback_window = max(
        feature_window_validation,
        key=lambda row: max(item["average_precision"] for item in row.get("model_results", [{"average_precision": 0.0}])),
    )
    best_prediction_model_name, best_prediction_model_ap = _best_model_by_ap(best_prediction_window["model_results"])
    best_lookback_model_name, best_lookback_model_ap = _best_model_by_ap(best_lookback_window["model_results"])
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
        "input_csv": str(feature_path),
        "events_csv": str(events_path),
        "split_strategy": split_description,
        "spend_cap_quantile": args.spend_cap_quantile,
        "calibration_method": args.calibration_method,
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
