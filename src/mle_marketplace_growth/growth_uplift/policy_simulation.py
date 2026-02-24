"""Run budget-constrained targeting simulation from uplift scores and feature proxies.

This module is a business-policy simulation layer (not causal ground truth).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_csv_by_key(path: Path) -> dict[tuple[str, str], dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in CSV: {path}")
    keyed_rows = {}
    for row in rows:
        keyed_rows[(row["user_id"], row["as_of_date"])] = row
    return keyed_rows


def _purchase_probability_proxy(frequency_90d: float) -> float:
    return max(0.0, min(1.0, frequency_90d / (frequency_90d + 3.0)))


def _write_plot(selected: list[dict], cost_per_user: float, output_path: Path) -> None:
    if not selected:
        raise ValueError("No selected users to plot. Check budget/cost configuration.")

    cumulative_spend = []
    cumulative_lift = []
    spend = 0.0
    lift = 0.0
    for row in selected:
        spend += cost_per_user
        lift += row["expected_incremental_value"]
        cumulative_spend.append(spend)
        cumulative_lift.append(lift)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cumulative_spend, cumulative_lift, color="tab:green", linewidth=2)
    ax.set_xlabel("Budget spend")
    ax.set_ylabel("Cumulative expected revenue lift (proxy)")
    ax.set_title("Policy Simulation: Lift vs Budget (Proxy)")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote policy simulation plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-constrained uplift policy simulation.")
    parser.add_argument(
        "--features-csv",
        default="data/gold/feature_store/growth_uplift/user_features_asof/as_of_date=2011-11-09/user_features_asof.csv",
        help="Path to user feature snapshot CSV",
    )
    parser.add_argument(
        "--scores-csv",
        default="artifacts/growth_uplift/prediction_scores.csv",
        help="Path to prediction scores CSV",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/growth_uplift/policy_simulation.json",
        help="Path to policy simulation output JSON",
    )
    parser.add_argument(
        "--output-plot",
        default="artifacts/growth_uplift/policy_simulation_budget_curve.png",
        help="Optional output PNG path for budget-vs-lift proxy curve",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=5000.0,
        help="Total budget for targeting simulation",
    )
    parser.add_argument(
        "--cost-per-user",
        type=float,
        default=5.0,
        help="Assumed treatment cost per targeted user",
    )
    parser.add_argument(
        "--margin-rate",
        type=float,
        default=0.35,
        help="Assumed gross margin rate applied to revenue proxy",
    )
    args = parser.parse_args()

    if args.budget <= 0.0:
        raise ValueError("--budget must be > 0")
    if args.cost_per_user <= 0.0:
        raise ValueError("--cost-per-user must be > 0")
    if not 0.0 < args.margin_rate <= 1.0:
        raise ValueError("--margin-rate must be in (0, 1]")

    features_path = Path(args.features_csv)
    scores_path = Path(args.scores_csv)
    if not features_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_path}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_path}")

    features_by_key = _load_csv_by_key(features_path)
    scores_by_key = _load_csv_by_key(scores_path)
    shared_keys = sorted(set(features_by_key.keys()) & set(scores_by_key.keys()))
    if not shared_keys:
        raise ValueError("No overlapping user_id/as_of_date keys between features and scores.")

    max_positive_uplift = max(0.0, max(float(scores_by_key[key]["uplift_score"]) for key in shared_keys))
    users = []
    for key in shared_keys:
        feature_row = features_by_key[key]
        score_row = scores_by_key[key]
        uplift_score = float(score_row["uplift_score"])
        responsiveness_proxy = (max(0.0, uplift_score) / max_positive_uplift) if max_positive_uplift > 0 else 0.0
        purchase_probability_proxy = _purchase_probability_proxy(float(feature_row["frequency_90d"]))
        expected_margin_proxy = max(0.0, float(feature_row["avg_basket_value_90d"])) * args.margin_rate
        expected_incremental_value = purchase_probability_proxy * expected_margin_proxy * responsiveness_proxy
        users.append(
            {
                "user_id": key[0],
                "as_of_date": key[1],
                "uplift_score": uplift_score,
                "purchase_probability_proxy": purchase_probability_proxy,
                "expected_margin_proxy": expected_margin_proxy,
                "responsiveness_proxy": responsiveness_proxy,
                "expected_incremental_value": expected_incremental_value,
            }
        )

    users.sort(key=lambda row: row["expected_incremental_value"], reverse=True)
    max_target_count = int(args.budget // args.cost_per_user)
    selected = users[:max_target_count]
    total_expected_lift = sum(row["expected_incremental_value"] for row in selected)
    spend = len(selected) * args.cost_per_user

    output = {
        "scope": "policy_simulation_from_estimated_response_proxies_not_causal_ground_truth",
        "inputs": {
            "features_csv": str(features_path),
            "scores_csv": str(scores_path),
        },
        "assumptions": {
            "budget": args.budget,
            "cost_per_user": args.cost_per_user,
            "margin_rate": args.margin_rate,
            "purchase_probability_proxy": "frequency_90d / (frequency_90d + 3)",
            "responsiveness_proxy": "max(0, uplift_score) normalized by max positive uplift_score",
        },
        "selection": {
            "candidate_users": len(users),
            "targeted_users": len(selected),
            "budget_spend": round(spend, 4),
            "budget_unused": round(args.budget - spend, 4),
        },
        "kpis": {
            "expected_revenue_lift_proxy_total": round(total_expected_lift, 6),
            "budget_efficiency_lift_per_dollar": round(total_expected_lift / spend, 9) if spend > 0 else 0.0,
        },
        "top_targets_preview": [
            {
                "user_id": row["user_id"],
                "as_of_date": row["as_of_date"],
                "expected_incremental_value": round(row["expected_incremental_value"], 6),
            }
            for row in selected[:20]
        ],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote policy simulation: {output_path}")
    _write_plot(selected, args.cost_per_user, Path(args.output_plot))


if __name__ == "__main__":
    main()
