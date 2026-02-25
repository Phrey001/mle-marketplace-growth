"""Run budget-constrained offline policy evaluation from expected-value scores."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


# ===== Data Loading =====
def _load_scores(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in scored CSV: {path}")
    required_columns = {"user_id", "as_of_date", "propensity_score", "expected_value_score"}
    missing_columns = sorted(required_columns - set(rows[0].keys()))
    if missing_columns:
        raise ValueError(f"Missing columns in scored CSV: {missing_columns}")
    return rows


# ===== Visualization =====
def _write_budget_curve(selected_rows: list[dict], cost_per_user: float, output_path: Path) -> None:
    if not selected_rows:
        raise ValueError("No selected users to plot.")
    cumulative_budget = []
    cumulative_expected_value = []
    budget_spend = 0.0
    expected_value = 0.0
    for row in selected_rows:
        budget_spend += cost_per_user
        expected_value += float(row["expected_value_score"])
        cumulative_budget.append(budget_spend)
        cumulative_expected_value.append(expected_value)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(cumulative_budget, cumulative_expected_value, linewidth=2, color="tab:blue")
    axis.set_xlabel("Budget spend")
    axis.set_ylabel("Cumulative expected value")
    axis.set_title("Incentive Allocation Curve")
    axis.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote offline policy evaluation plot: {output_path}")


# ===== Entry Point =====
def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Budget-constrained offline policy evaluation from expected value scores.")
    parser.add_argument("--scores-csv", default="artifacts/purchase_propensity/prediction_scores.csv", help="Path to scored users CSV from predict.py")
    parser.add_argument("--output-json", default="artifacts/purchase_propensity/offline_policy_evaluation.json", help="Path to output offline policy evaluation summary")
    parser.add_argument("--output-plot", default="artifacts/purchase_propensity/offline_policy_evaluation_budget_curve.png", help="Path to output budget curve PNG")
    parser.add_argument("--budget", type=float, default=5000.0, help="Total targeting budget")
    parser.add_argument("--cost-per-user", type=float, default=5.0, help="Assumed cost per targeted user")
    args = parser.parse_args()

    # ===== Argument Checks =====
    if args.budget <= 0.0:
        raise ValueError("--budget must be greater than 0")
    if args.cost_per_user <= 0.0:
        raise ValueError("--cost-per-user must be greater than 0")

    # ===== Select Users + Compute KPIs =====
    rows = _load_scores(Path(args.scores_csv))
    rows.sort(key=lambda row: float(row["expected_value_score"]), reverse=True)
    target_count = int(args.budget // args.cost_per_user)
    selected_rows = rows[:target_count]
    if not selected_rows:
        raise ValueError("Budget is too small to target any user.")

    budget_spend = len(selected_rows) * args.cost_per_user
    expected_value_total = sum(float(row["expected_value_score"]) for row in selected_rows)
    output = {
        "scope": "offline_policy_evaluation_not_causal_promotional_incrementality",
        "inputs": {"scores_csv": args.scores_csv},
        "assumptions": {
            "budget": args.budget,
            "cost_per_user": args.cost_per_user,
            "allocation_rule": "target highest expected_value_score until budget exhausted",
        },
        "selection": {
            "candidate_users": len(rows),
            "targeted_users": len(selected_rows),
            "budget_spend": round(budget_spend, 6),
            "budget_unused": round(args.budget - budget_spend, 6),
        },
        "kpis": {
            "expected_value_total": round(expected_value_total, 6),
            "expected_value_per_targeted_user": round(expected_value_total / len(selected_rows), 6),
            "expected_value_per_dollar": round(expected_value_total / budget_spend, 9),
        },
        "top_targets_preview": [
            {
                "user_id": row["user_id"],
                "as_of_date": row["as_of_date"],
                "propensity_score": round(float(row["propensity_score"]), 6),
                "expected_value_score": round(float(row["expected_value_score"]), 6),
            }
            for row in selected_rows[:20]
        ],
    }

    # ===== Write Outputs =====
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote offline policy evaluation: {output_path}")
    _write_budget_curve(selected_rows, args.cost_per_user, Path(args.output_plot))


if __name__ == "__main__":
    main()
