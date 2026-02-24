"""Train a baseline uplift model from feature-store growth-uplift data.

Mechanical flow:
1) Simulate treated/control assignment with deterministic hash bucketing.
2) Train two outcome models (treated and control arms):
   - treated model is fit only on treated rows and learns "expected outcome if treated"
   - control model is fit only on control rows and learns "expected outcome if not treated"
3) Compute uplift score = treated prediction - control prediction.
4) Evaluate policy lift on validation and keep the best baseline model.

Scope note:
- Treatment assignment here is simulated (`treatment` via hash bucketing).
- Outputs are for offline model-logic validation, not direct business-effect claims.
- Causal business impact is not identified here because treatment is synthetic.
"""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from .evaluate import build_policy_table


NUMERIC_FEATURES = [
    "recency_days",
    "frequency_30d",
    "frequency_90d",
    "monetary_30d",
    "monetary_90d",
    "avg_basket_value_90d",
]


def _stable_ratio(key: str) -> float:
    """Map a key deterministically to [0, 1) so assignment is repeatable across reruns."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    return value / float(0xFFFFFFFFFFFFFFFF)


def _assign_treatment(user_id: str, as_of_date: str, treatment_rate: float) -> int:
    """Simulate A/B treatment with stable hash buckets instead of true randomness.

    Why: this creates a reproducible pseudo-random treated/control split for offline
    experiments, so the same user/date always lands in the same bucket.
    """
    return 1 if _stable_ratio(f"{user_id}|{as_of_date}|treatment") < treatment_rate else 0


def _is_validation(user_id: str, as_of_date: str, validation_rate: float) -> bool:
    """Create a stable train/validation split with hash bucketing.

    Why: avoids split drift between reruns, making model comparisons fair.
    """
    return _stable_ratio(f"{user_id}|{as_of_date}|split") < validation_rate


def _load_training_rows(input_path: Path, treatment_rate: float, label_column: str) -> list[dict]:
    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            user_id = row["user_id"]
            as_of_date = row["as_of_date"]
            features = {feature: float(row[feature]) for feature in NUMERIC_FEATURES}
            features["country"] = row["country"]
            outcome = float(row[label_column])
            treatment = _assign_treatment(user_id, as_of_date, treatment_rate)
            rows.append(
                {
                    "user_id": user_id,
                    "as_of_date": as_of_date,
                    "features": features,
                    "outcome": outcome,
                    "treatment": treatment,
                }
            )
    if not rows:
        raise ValueError(f"No rows found in input dataset: {input_path}")
    return rows


def _subset(rows: list[dict], treatment_value: int) -> tuple[list[dict], list[float]]:
    feature_rows = [row["features"] for row in rows if row["treatment"] == treatment_value]
    outcomes = [row["outcome"] for row in rows if row["treatment"] == treatment_value]
    return feature_rows, outcomes


def _evaluate_uplift_predictions(val_rows: list[dict], uplift_pred: list[float]) -> dict:
    evaluation_rows = []
    for row, uplift in zip(val_rows, uplift_pred, strict=True):
        evaluation_rows.append(
            {
                "user_id": row["user_id"],
                "as_of_date": row["as_of_date"],
                "treatment": row["treatment"],
                "observed_outcome": row["outcome"],
                "uplift_score": float(uplift),
            }
        )
    return build_policy_table(evaluation_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline growth uplift model.")
    parser.add_argument(
        "--input-csv",
        default="data/gold/feature_store/growth_uplift/uplift_train_dataset/as_of_date=2011-12-09/uplift_train_dataset.csv",
        help="Path to growth uplift training dataset CSV",
    )
    parser.add_argument(
        "--label-column",
        default="label_net_revenue_30d",
        help="Label column to model",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/growth_uplift",
        help="Output directory for model and metrics",
    )
    parser.add_argument(
        "--treatment-rate",
        type=float,
        default=0.5,
        help="Deterministic pseudo-random treatment allocation rate",
    )
    parser.add_argument(
        "--validation-rate",
        type=float,
        default=0.2,
        help="Deterministic validation split fraction",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for regressors",
    )
    args = parser.parse_args()
    print(
        "WARNING: simulation mode uses synthetic treatment assignment; "
        "results are for model-logic validation, not causal business impact."
    )

    if not 0.0 < args.treatment_rate < 1.0:
        raise ValueError("--treatment-rate must be between 0 and 1")
    if not 0.0 < args.validation_rate < 1.0:
        raise ValueError("--validation-rate must be between 0 and 1")

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    rows = _load_training_rows(input_path, args.treatment_rate, args.label_column)
    train_rows = [row for row in rows if not _is_validation(row["user_id"], row["as_of_date"], args.validation_rate)]
    val_rows = [row for row in rows if _is_validation(row["user_id"], row["as_of_date"], args.validation_rate)]
    if not train_rows or not val_rows:
        raise ValueError("Train/validation split produced an empty subset; adjust --validation-rate.")

    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit([row["features"] for row in train_rows])

    treated_features, treated_outcomes = _subset(train_rows, treatment_value=1)
    control_features, control_outcomes = _subset(train_rows, treatment_value=0)
    if not treated_features or not control_features:
        raise ValueError("Need both treated and control rows; adjust --treatment-rate.")

    candidate_builders = [
        (
            "random_forest",
            lambda seed: (
                RandomForestRegressor(n_estimators=200, random_state=seed, min_samples_leaf=20),
                RandomForestRegressor(n_estimators=200, random_state=seed + 1, min_samples_leaf=20),
            ),
        ),
        (
            "gradient_boosting",
            lambda seed: (
                GradientBoostingRegressor(random_state=seed),
                GradientBoostingRegressor(random_state=seed + 1),
            ),
        ),
    ]

    val_matrix = vectorizer.transform([row["features"] for row in val_rows])
    candidate_results = []
    for model_name, model_builder in candidate_builders:
        treated_model, control_model = model_builder(args.random_seed)
        # Treated arm model learns outcome from treated-only rows.
        treated_model.fit(vectorizer.transform(treated_features), treated_outcomes)
        # Control arm model learns outcome from control-only rows.
        control_model.fit(vectorizer.transform(control_features), control_outcomes)
        treated_pred = treated_model.predict(val_matrix)
        control_pred = control_model.predict(val_matrix)
        # Per user uplift estimate: expected treated outcome minus expected control outcome.
        uplift_pred = treated_pred - control_pred
        evaluation = _evaluate_uplift_predictions(val_rows, uplift_pred)
        top_delta = evaluation["interpretation_kpis"]["top10_vs_all_users_lift_delta"]
        candidate_results.append(
            {
                "model_name": model_name,
                "treated_model": treated_model,
                "control_model": control_model,
                "treated_pred": treated_pred,
                "control_pred": control_pred,
                "uplift_pred": uplift_pred,
                "evaluation": evaluation,
                "selection_kpi_top10_vs_all_users_lift_delta": float(top_delta),
            }
        )

    best_result = max(candidate_results, key=lambda result: result["selection_kpi_top10_vs_all_users_lift_delta"])
    selected_model_name = best_result["model_name"]
    treated_model = best_result["treated_model"]
    control_model = best_result["control_model"]
    treated_pred = best_result["treated_pred"]
    control_pred = best_result["control_pred"]
    uplift_pred = best_result["uplift_pred"]
    evaluation = best_result["evaluation"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "uplift_model.pkl"
    metrics_path = output_dir / "train_metrics.json"
    scores_path = output_dir / "validation_scores.csv"

    model_bundle = {
        "vectorizer": vectorizer,
        "treated_model": treated_model,
        "control_model": control_model,
        "model_name": selected_model_name,
        "feature_columns": NUMERIC_FEATURES + ["country"],
        "label_column": args.label_column,
        "treatment_rate": args.treatment_rate,
    }
    with model_path.open("wb") as file:
        pickle.dump(model_bundle, file)

    metrics = {
        "input_csv": str(input_path),
        "label_column": args.label_column,
        "treatment_rate": args.treatment_rate,
        "validation_rate": args.validation_rate,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "treated_train_rows": len(treated_features),
        "control_train_rows": len(control_features),
        "selected_model_name": selected_model_name,
        "model_candidates": [
            {
                "model_name": result["model_name"],
                "selection_kpi_top10_vs_all_users_lift_delta": round(
                    result["selection_kpi_top10_vs_all_users_lift_delta"], 6
                ),
                "sanity_status": result["evaluation"]["sanity"]["status"],
                "sanity_flags": result["evaluation"]["sanity"]["flags"],
            }
            for result in candidate_results
        ],
        "validation_policy_lift": evaluation["policy_lift"],
        "validation_inferred_treatment_rate": evaluation["inferred_treatment_rate"],
        "validation_interpretation_kpis": evaluation["interpretation_kpis"],
        "validation_sanity": evaluation["sanity"],
        "training_scope": "simulation_only_not_direct_business_validation",
        "causal_validity": "not_identified_without_real_treatment_logs",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    with scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "user_id",
                "as_of_date",
                "treatment",
                "observed_outcome",
                "pred_treated",
                "pred_control",
                "uplift_score",
            ]
        )
        for row, pred_t, pred_c, uplift in zip(val_rows, treated_pred, control_pred, uplift_pred, strict=True):
            writer.writerow(
                [
                    row["user_id"],
                    row["as_of_date"],
                    row["treatment"],
                    round(float(row["outcome"]), 6),
                    round(float(pred_t), 6),
                    round(float(pred_c), 6),
                    round(float(uplift), 6),
                ]
            )

    print(f"Wrote model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote validation scores: {scores_path}")
    print(f"Selected model: {selected_model_name}")


if __name__ == "__main__":
    main()
