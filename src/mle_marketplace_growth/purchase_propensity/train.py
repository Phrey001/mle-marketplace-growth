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
import pickle
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

FEATURE_COLUMNS = [
    "recency_days",
    "frequency_30d",
    "frequency_90d",
    "monetary_30d",
    "monetary_90d",
    "avg_basket_value_90d",
]


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


def _apply_spend_cap(rows: list[dict], spend_cap_value: float) -> None:
    for row in rows:
        row["features"]["monetary_90d"] = min(row["features"]["monetary_90d"], spend_cap_value)


def _stable_ratio(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    return value / float(0xFFFFFFFFFFFFFFFF)


# ===== Data Preparation =====
def _load_training_rows(input_path: Path) -> list[dict]:
    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            features = {feature: float(row[feature]) for feature in FEATURE_COLUMNS}
            features["country"] = row["country"]
            purchase_label = float(row["label_purchase_30d"])
            revenue_label = float(row["label_net_revenue_30d"])
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
def _split_rows(rows: list[dict], validation_mode: str, validation_rate: float) -> tuple[list[dict], list[dict], str]:
    if validation_mode == "out_of_time":
        unique_dates = sorted({row["as_of_date"] for row in rows})
        if len(unique_dates) < 2:
            raise ValueError("Out-of-time validation needs at least 2 as_of_date values.")
        holdout_date_count = max(1, int(round(len(unique_dates) * validation_rate)))
        holdout_dates = set(unique_dates[-holdout_date_count:])
        train_rows = [row for row in rows if row["as_of_date"] not in holdout_dates]
        validation_rows = [row for row in rows if row["as_of_date"] in holdout_dates]
        split_desc = f"out_of_time_holdout_dates={sorted(holdout_dates)}"
        return train_rows, validation_rows, split_desc

    train_rows = []
    validation_rows = []
    for row in rows:
        split_ratio = _stable_ratio(f'{row["user_id"]}|{row["as_of_date"]}|split')
        if split_ratio < validation_rate:
            validation_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, validation_rows, "deterministic_hash_split"


def _policy_metrics(rows: list[dict], policy_name: str, selected_mask: list[bool]) -> dict:
    selected_rows = [row for row, is_selected in zip(rows, selected_mask, strict=True) if is_selected]
    if not selected_rows:
        raise ValueError(f"Policy {policy_name} selected zero users.")
    purchase_rate = sum(row["purchase_label"] for row in selected_rows) / len(selected_rows)
    realized_revenue = sum(row["revenue_label"] for row in selected_rows)
    return {
        "policy": policy_name,
        "targeted_users": len(selected_rows),
        "purchase_rate": round(purchase_rate, 6),
        "actual_revenue_total": round(realized_revenue, 6),
        "actual_revenue_per_targeted_user": round(realized_revenue / len(selected_rows), 6),
    }


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


# ===== Entry Point =====
def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Train propensity scoring models and backtest targeting policies.")
    parser.add_argument("--input-csv", default="data/gold/feature_store/purchase_propensity/propensity_train_dataset/as_of_date=2011-11-09/propensity_train_dataset.csv", help="Path to training dataset CSV")
    parser.add_argument("--output-dir", default="artifacts/purchase_propensity", help="Output directory for model and metrics")
    parser.add_argument("--target-rate", type=float, default=0.2, help="Share of users targeted in policy backtest")
    parser.add_argument("--validation-rate", type=float, default=0.2, help="Holdout fraction used for validation split")
    parser.add_argument(
        "--validation-mode",
        choices=["hash", "out_of_time"],
        default="hash",
        help="Validation split strategy. Use out_of_time when multiple as_of_date snapshots are present.",
    )
    parser.add_argument("--spend-cap-quantile", type=float, default=0.99, help="Quantile cap for monetary_90d to reduce extreme-spend dominance (0 < q <= 1).")
    parser.add_argument(
        "--calibration-method",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration method applied on train folds before validation scoring.",
    )
    args = parser.parse_args()

    # ===== Input Checks =====
    if not 0.0 < args.target_rate < 1.0:
        raise ValueError("--target-rate must be between 0 and 1")
    if not 0.0 < args.validation_rate < 1.0:
        raise ValueError("--validation-rate must be between 0 and 1")
    if not 0.0 < args.spend_cap_quantile <= 1.0:
        raise ValueError("--spend-cap-quantile must be in (0, 1]")

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # ===== Load Data + Split =====
    rows = _load_training_rows(input_path)
    train_rows, validation_rows, split_description = _split_rows(rows, args.validation_mode, args.validation_rate)
    if not train_rows or not validation_rows:
        raise ValueError("Train/validation split produced an empty subset.")

    train_labels = [int(row["purchase_label"]) for row in train_rows]
    if len(set(train_labels)) < 2:
        raise ValueError("Training labels contain only one class. Try an earlier as_of_date.")

    spend_cap_value = _quantile(
        [float(row["features"]["monetary_90d"]) for row in train_rows],
        args.spend_cap_quantile,
    )
    _apply_spend_cap(train_rows, spend_cap_value)
    _apply_spend_cap(validation_rows, spend_cap_value)

    # ===== Train Model Candidates =====
    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit([row["features"] for row in train_rows])
    train_matrix = vectorizer.transform([row["features"] for row in train_rows])
    validation_matrix = vectorizer.transform([row["features"] for row in validation_rows])

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

    candidate_results = []
    validation_labels = [int(row["purchase_label"]) for row in validation_rows]
    for model_name, model in model_candidates:
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
                "model": calibrated_model,
                "propensity_scores": propensity_scores,
                "roc_auc": float(roc_auc),
                "average_precision": float(avg_precision),
            }
        )

    # ===== Train Revenue Model (Conditional on Purchase) =====
    positive_train_indices = [index for index, label in enumerate(train_labels) if label == 1]
    if not positive_train_indices:
        raise ValueError("No positive purchase rows for conditional revenue training.")
    positive_revenue_targets = [float(train_rows[index]["revenue_label"]) for index in positive_train_indices]
    revenue_fallback_value = sum(positive_revenue_targets) / len(positive_revenue_targets)
    revenue_model = None
    if len(positive_train_indices) >= 2:
        positive_train_matrix = train_matrix[positive_train_indices]
        revenue_model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        revenue_model.fit(positive_train_matrix, positive_revenue_targets)

    best_result = max(candidate_results, key=lambda result: result["average_precision"])
    selected_model_name = best_result["model_name"]
    selected_model = best_result["model"]
    propensity_scores = best_result["propensity_scores"]
    selected_roc_auc = float(best_result["roc_auc"])
    selected_average_precision = float(best_result["average_precision"])
    if revenue_model is None:
        predicted_conditional_revenue = [max(0.0, revenue_fallback_value) for _ in validation_rows]
        revenue_model_name = "constant_mean_positive_revenue"
    else:
        predicted_conditional_revenue = [max(0.0, float(value)) for value in revenue_model.predict(validation_matrix)]
        revenue_model_name = "xgboost_regressor_conditional_revenue"

    base_positive_rate = sum(validation_labels) / len(validation_labels)
    top_decile_count = max(1, int(0.1 * len(validation_labels)))
    top_decile_indices = sorted(
        range(len(validation_rows)),
        key=lambda index: propensity_scores[index],
        reverse=True,
    )[:top_decile_count]
    top_decile_positive_rate = sum(validation_labels[index] for index in top_decile_indices) / top_decile_count
    top_decile_lift = top_decile_positive_rate / base_positive_rate if base_positive_rate > 0 else 0.0
    brier_score = float(brier_score_loss(validation_labels, propensity_scores))
    ece_10_bin = _expected_calibration_error(validation_labels, propensity_scores.tolist(), bins=10)

    # ===== Holdout Policy Backtest =====
    expected_value_scores = [
        float(score) * float(revenue)
        for score, revenue in zip(propensity_scores, predicted_conditional_revenue, strict=True)
    ]
    ranked_indices = sorted(range(len(validation_rows)), key=lambda index: expected_value_scores[index], reverse=True)
    target_count = max(1, int(len(validation_rows) * args.target_rate))
    ml_selected = {index for index in ranked_indices[:target_count]}
    random_ranked_indices = sorted(range(len(validation_rows)), key=lambda index: _stable_ratio(f'{validation_rows[index]["user_id"]}|{validation_rows[index]["as_of_date"]}|policy_random'))
    random_selected = {index for index in random_ranked_indices[:target_count]}
    rfm_scores = [
        (1.0 / (1.0 + row["features"]["recency_days"]))
        + 0.5 * row["features"]["frequency_90d"]
        + 0.01 * row["features"]["monetary_90d"]
        for row in validation_rows
    ]
    rfm_ranked_indices = sorted(range(len(validation_rows)), key=lambda index: rfm_scores[index], reverse=True)
    rfm_selected = {index for index in rfm_ranked_indices[:target_count]}
    row_count = len(validation_rows)

    policy_comparison = [
        _policy_metrics(validation_rows, "ml_top_expected_value", [index in ml_selected for index in range(row_count)]),
        _policy_metrics(validation_rows, "random_baseline", [index in random_selected for index in range(row_count)]),
        _policy_metrics(validation_rows, "rfm_heuristic", [index in rfm_selected for index in range(row_count)]),
    ]

    # ===== Write Artifacts =====
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "propensity_model.pkl"
    metrics_path = output_dir / "train_metrics.json"
    scores_path = output_dir / "validation_predictions.csv"

    model_bundle = {
        "vectorizer": vectorizer,
        "propensity_model": selected_model,
        "propensity_model_name": selected_model_name,
        "revenue_model": revenue_model,
        "revenue_model_name": revenue_model_name,
        "revenue_fallback_value": round(float(revenue_fallback_value), 6),
        "feature_columns": FEATURE_COLUMNS,
        "spend_cap_value": float(spend_cap_value),
        "spend_cap_quantile": float(args.spend_cap_quantile),
        "calibration_method": args.calibration_method,
        "target_definition": "purchase_in_next_30_days",
        "expected_value_definition": "propensity_score * predicted_conditional_revenue_30d",
    }
    with model_path.open("wb") as file:
        pickle.dump(model_bundle, file)

    metrics = {
        "input_csv": str(input_path),
        "target_definition": "label_purchase_30d",
        "revenue_backtest_label": "label_net_revenue_30d",
        "validation_mode": args.validation_mode,
        "validation_split_description": split_description,
        "target_rate": args.target_rate,
        "validation_rate": args.validation_rate,
        "spend_cap_quantile": args.spend_cap_quantile,
        "spend_cap_value": round(float(spend_cap_value), 6),
        "calibration_method": args.calibration_method,
        "revenue_model_name": revenue_model_name,
        "revenue_fallback_value": round(float(revenue_fallback_value), 6),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "selected_model_name": selected_model_name,
        "validation_quality": {
            "roc_auc": round(selected_roc_auc, 6),
            "average_precision": round(selected_average_precision, 6),
            "top_decile_lift": round(float(top_decile_lift), 6),
            "top_decile_positive_rate": round(float(top_decile_positive_rate), 6),
            "base_positive_rate": round(float(base_positive_rate), 6),
            "brier_score": round(brier_score, 6),
            "ece_10_bin": round(ece_10_bin, 6),
        },
        "model_candidates": [
            {
                "model_name": result["model_name"],
                "roc_auc": round(result["roc_auc"], 6),
                "average_precision": round(result["average_precision"], 6),
            }
            for result in candidate_results
        ],
        "offline_policy_backtest": {
            "target_rate": args.target_rate,
            "policies": policy_comparison,
        },
        "scope": "offline_policy_backtest_not_causal_promotional_incrementality",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    with scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "user_id",
                "as_of_date",
                "label_purchase_30d",
                "label_net_revenue_30d",
                "propensity_score",
                "predicted_conditional_revenue_30d",
                "expected_value_score",
                "is_selected_ml_policy",
                "is_selected_random_policy",
                "is_selected_rfm_policy",
            ]
        )
        for index, (row, propensity, expected_value) in enumerate(
            zip(validation_rows, propensity_scores, expected_value_scores, strict=True)
        ):
            writer.writerow(
                [
                    row["user_id"],
                    row["as_of_date"],
                    int(row["purchase_label"]),
                    round(float(row["revenue_label"]), 6),
                    round(float(propensity), 6),
                    round(float(predicted_conditional_revenue[index]), 6),
                    round(float(expected_value), 6),
                    1 if index in ml_selected else 0,
                    1 if index in random_selected else 0,
                    1 if index in rfm_selected else 0,
                ]
            )

    # ===== Run Summary =====
    print(f"Wrote model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote validation predictions: {scores_path}")
    print(f"Selected model: {selected_model_name}")


if __name__ == "__main__":
    main()
