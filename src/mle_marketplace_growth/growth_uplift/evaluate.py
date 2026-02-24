"""Evaluate simulated uplift validation scores for model-logic checks.

This evaluator is for offline simulation diagnostics, not direct business validation.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_scores(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in validation scores: {path}")
    required = {"user_id", "as_of_date", "treatment", "observed_outcome", "uplift_score"}
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise ValueError(f"Missing required columns in validation scores: {missing}")
    return rows


def build_policy_table(rows: list[dict]) -> dict:
    """Compute policy lift with inverse-propensity-style reweighting.

    Layman intuition:
    - Treated and control group sizes are usually not perfectly equal.
    - Reweighting by treatment probability puts both groups on a comparable scale.
    - This gives a fairer estimate of incremental lift for a candidate policy.

    Concrete intuition:
    - If only ~30% are treated, each treated outcome represents about 1 / 0.30 ~= 3.3
      users in a balanced view.
    - If ~70% are control, each control outcome represents about 1 / 0.70 ~= 1.4
      users in that same balanced view.
    - Estimated incremental lift is then:
      (reweighted treated outcomes) - (reweighted control outcomes).
    """
    parsed = [
        {
            "user_id": row.get("user_id", ""),
            "as_of_date": row.get("as_of_date", ""),
            "treatment": int(row["treatment"]),
            "observed_outcome": float(row["observed_outcome"]),
            "uplift_score": float(row["uplift_score"]),
        }
        for row in rows
    ]
    parsed.sort(key=lambda row: row["uplift_score"], reverse=True)

    treatment_rate = sum(row["treatment"] for row in parsed) / len(parsed)
    if treatment_rate <= 0.0 or treatment_rate >= 1.0:
        raise ValueError("Treatment rate inferred from validation scores must be between 0 and 1.")

    fractions = [0.1, 0.2, 0.3, 0.5, 1.0]
    policies = []
    for fraction in fractions:
        selected_count = max(1, int(len(parsed) * fraction))
        selected = parsed[:selected_count]
        total_lift = 0.0
        for row in selected:
            # Step 1: expand each observed outcome by how rare/common its group is.
            # Step 2: subtract control contribution from treated contribution.
            # Result: estimated incremental value relative to "no treatment uplift".
            if row["treatment"] == 1:
                total_lift += row["observed_outcome"] / treatment_rate
            else:
                total_lift -= row["observed_outcome"] / (1.0 - treatment_rate)
        policies.append(
            {
                "top_fraction": fraction,
                "selected_users": selected_count,
                "estimated_incremental_net_revenue_total": round(total_lift, 4),
                "estimated_incremental_net_revenue_per_user": round(total_lift / selected_count, 6),
            }
        )

    top_10 = next(row for row in policies if row["top_fraction"] == 0.1)
    all_users = next(row for row in policies if row["top_fraction"] == 1.0)
    nonzero_outcome_rows = sum(1 for row in parsed if row["observed_outcome"] != 0.0)
    outcome_sum = sum(row["observed_outcome"] for row in parsed)
    top10_vs_full_per_user_delta = (
        top_10["estimated_incremental_net_revenue_per_user"]
        - all_users["estimated_incremental_net_revenue_per_user"]
    )

    sanity_flags = []
    if nonzero_outcome_rows == 0:
        sanity_flags.append("all_outcomes_zero_check_as_of_date_window")
    if not 0.2 <= treatment_rate <= 0.8:
        sanity_flags.append("treatment_rate_imbalance")
    if abs(all_users["estimated_incremental_net_revenue_per_user"]) < 1e-9 and abs(top_10["estimated_incremental_net_revenue_per_user"]) < 1e-9:
        sanity_flags.append("policy_lift_all_zero")
    elif top10_vs_full_per_user_delta <= 0:
        sanity_flags.append("no_top_bucket_lift_improvement")

    return {
        "validation_rows": len(parsed),
        "inferred_treatment_rate": round(treatment_rate, 6),
        "outcome_summary": {
            "nonzero_outcome_rows": nonzero_outcome_rows,
            "outcome_sum": round(outcome_sum, 6),
        },
        "interpretation_kpis": {
            "top_10_percent_lift_per_user": top_10["estimated_incremental_net_revenue_per_user"],
            "all_users_lift_per_user": all_users["estimated_incremental_net_revenue_per_user"],
            "top10_vs_all_users_lift_delta": round(top10_vs_full_per_user_delta, 6),
        },
        "sanity": {
            "status": "ok" if not sanity_flags else "review",
            "flags": sanity_flags,
        },
        "policy_lift": policies,
    }


def _write_plot(evaluation: dict, output_path: Path) -> None:
    fractions = [row["top_fraction"] for row in evaluation["policy_lift"]]
    per_user_lift = [row["estimated_incremental_net_revenue_per_user"] for row in evaluation["policy_lift"]]
    total_lift = [row["estimated_incremental_net_revenue_total"] for row in evaluation["policy_lift"]]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(fractions, per_user_lift, marker="o", label="Lift per user")
    ax1.set_xlabel("Top fraction treated by uplift score")
    ax1.set_ylabel("Estimated incremental net revenue per user")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(fractions, total_lift, marker="s", linestyle="--", label="Total lift", color="tab:orange")
    ax2.set_ylabel("Estimated incremental net revenue total")

    ax1.set_title("Growth Uplift Policy Lift Summary (Simulation)")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote evaluation plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate uplift validation scores.")
    parser.add_argument(
        "--scores-csv",
        default="artifacts/growth_uplift/validation_scores.csv",
        help="Path to validation score CSV emitted by training",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/growth_uplift/evaluation.json",
        help="Path to compact evaluation output JSON",
    )
    parser.add_argument(
        "--output-plot",
        default="artifacts/growth_uplift/evaluation_policy_lift.png",
        help="Optional output PNG path for policy-lift plot",
    )
    args = parser.parse_args()
    print(
        "WARNING: evaluation is simulation-only (synthetic treatment); "
        "do not interpret as causal business impact."
    )

    scores_path = Path(args.scores_csv)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_path}")

    evaluation = build_policy_table(_load_scores(scores_path))
    evaluation["evaluation_scope"] = "simulation_only_not_direct_business_validation"
    evaluation["causal_validity"] = "not_identified_without_real_treatment_logs"
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote evaluation: {output_path}")
    _write_plot(evaluation, Path(args.output_plot))


if __name__ == "__main__":
    main()
