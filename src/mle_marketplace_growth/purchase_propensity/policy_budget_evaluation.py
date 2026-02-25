"""Run budget-constrained offline policy evaluation for ML/Random/RFM policies."""

import argparse
import csv
import json
from pathlib import Path

POLICIES = [
    ("ml_top_expected_value", "expected_value_score"),
    ("random_baseline", "random_policy_score"),
    ("rfm_heuristic", "rfm_policy_score"),
]


def _load_rows(path: Path, purchase_label_col: str, revenue_label_col: str) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in scores CSV: {path}")
    required_columns = {
        "user_id",
        "as_of_date",
        "expected_value_score",
        "random_policy_score",
        "rfm_policy_score",
        purchase_label_col,
        revenue_label_col,
    }
    missing_columns = sorted(required_columns - set(rows[0].keys()))
    if missing_columns:
        raise ValueError(f"Missing columns in scores CSV: {missing_columns}")
    return rows


def _policy_metrics(
    rows: list[dict],
    policy_name: str,
    score_col: str,
    target_count: int,
    purchase_label_col: str,
    revenue_label_col: str,
    cost_per_user: float,
) -> dict:
    ranked_rows = sorted(rows, key=lambda row: float(row[score_col]), reverse=True)
    selected_rows = ranked_rows[:target_count]
    if not selected_rows:
        raise ValueError(f"No selected rows for policy={policy_name} and target_count={target_count}")
    targeted_users = len(selected_rows)
    revenue_total = sum(float(row[revenue_label_col]) for row in selected_rows)
    purchase_rate = sum(float(row[purchase_label_col]) for row in selected_rows) / targeted_users
    return {
        "policy": policy_name,
        "targeted_users": targeted_users,
        "budget_spend": round(targeted_users * cost_per_user, 6),
        "actual_purchase_rate": round(purchase_rate, 6),
        "actual_revenue_total": round(revenue_total, 6),
        "actual_revenue_per_targeted_user": round(revenue_total / targeted_users, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-constrained policy comparison for ML/Random/RFM.")
    parser.add_argument("--scores-csv", required=True, help="Validation/Test predictions CSV from train.py")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    parser.add_argument("--budget", type=float, required=True, help="Total incentive budget")
    parser.add_argument("--cost-per-user", type=float, required=True, help="Cost per targeted user")
    parser.add_argument("--purchase-label-col", default="label_purchase_30d", help="Purchase label column")
    parser.add_argument("--revenue-label-col", default="label_net_revenue_30d", help="Revenue label column")
    args = parser.parse_args()

    if args.budget <= 0.0:
        raise ValueError("--budget must be greater than 0")
    if args.cost_per_user <= 0.0:
        raise ValueError("--cost-per-user must be greater than 0")

    rows = _load_rows(Path(args.scores_csv), args.purchase_label_col, args.revenue_label_col)
    target_count = int(args.budget // args.cost_per_user)
    if target_count < 1:
        raise ValueError("Budget is too small to target any user.")

    policy_comparison = [
        _policy_metrics(
            rows,
            policy_name=policy_name,
            score_col=score_col,
            target_count=target_count,
            purchase_label_col=args.purchase_label_col,
            revenue_label_col=args.revenue_label_col,
            cost_per_user=args.cost_per_user,
        )
        for policy_name, score_col in POLICIES
    ]

    by_policy = {row["policy"]: row for row in policy_comparison}
    output = {
        "scope": "offline_policy_budget_backtest_not_causal_promotional_incrementality",
        "inputs": {"scores_csv": args.scores_csv},
        "assumptions": {
            "budget": args.budget,
            "cost_per_user": args.cost_per_user,
            "target_count_by_budget": target_count,
            "policy_scoring_columns": {name: col for name, col in POLICIES},
        },
        "policy_comparison": policy_comparison,
        "interpretation_kpis": {
            "ml_vs_random_revenue_per_user_delta": round(
                by_policy["ml_top_expected_value"]["actual_revenue_per_targeted_user"]
                - by_policy["random_baseline"]["actual_revenue_per_targeted_user"],
                6,
            ),
            "ml_vs_rfm_revenue_per_user_delta": round(
                by_policy["ml_top_expected_value"]["actual_revenue_per_targeted_user"]
                - by_policy["rfm_heuristic"]["actual_revenue_per_targeted_user"],
                6,
            ),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote budget policy evaluation: {output_path}")


if __name__ == "__main__":
    main()
