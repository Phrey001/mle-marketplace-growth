"""Evaluate offline policy backtest results from validation predictions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

REQUIRED_COLUMNS = {
    "user_id",
    "as_of_date",
    "label_purchase_30d",
    "label_net_revenue_30d",
    "is_selected_ml_policy",
    "is_selected_random_policy",
    "is_selected_rfm_policy",
}
POLICY_COLUMNS = [
    ("ml_top_expected_value", "is_selected_ml_policy"),
    ("random_baseline", "is_selected_random_policy"),
    ("rfm_heuristic", "is_selected_rfm_policy"),
]


# ===== Data Loading =====
def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in validation predictions: {path}")
    missing_columns = sorted(REQUIRED_COLUMNS - set(rows[0].keys()))
    if missing_columns:
        raise ValueError(f"Missing columns in validation predictions: {missing_columns}")
    return rows


def _compute_policy_metrics(rows: list[dict], policy_name: str, selected_column: str) -> dict:
    selected_rows = [row for row in rows if int(row[selected_column]) == 1]
    if not selected_rows:
        raise ValueError(f"Policy {policy_name} selected zero rows.")

    targeted_users = len(selected_rows)
    purchase_rate = sum(float(row["label_purchase_30d"]) for row in selected_rows) / targeted_users
    revenue_total = sum(float(row["label_net_revenue_30d"]) for row in selected_rows)
    return {
        "policy": policy_name,
        "targeted_users": targeted_users,
        "purchase_rate": round(purchase_rate, 6),
        "actual_revenue_total": round(revenue_total, 6),
        "actual_revenue_per_targeted_user": round(revenue_total / targeted_users, 6),
    }


# ===== Metric Aggregation =====
def build_policy_backtest(rows: list[dict]) -> dict:
    policies = [_compute_policy_metrics(rows, policy_name, selected_column) for policy_name, selected_column in POLICY_COLUMNS]
    by_name = {row["policy"]: row for row in policies}
    ml_vs_random_revenue_delta = (
        by_name["ml_top_expected_value"]["actual_revenue_per_targeted_user"]
        - by_name["random_baseline"]["actual_revenue_per_targeted_user"]
    )
    ml_vs_rfm_revenue_delta = (
        by_name["ml_top_expected_value"]["actual_revenue_per_targeted_user"]
        - by_name["rfm_heuristic"]["actual_revenue_per_targeted_user"]
    )
    return {
        "validation_rows": len(rows),
        "policy_comparison": policies,
        "interpretation_kpis": {
            "ml_vs_random_revenue_per_user_delta": round(ml_vs_random_revenue_delta, 6),
            "ml_vs_rfm_revenue_per_user_delta": round(ml_vs_rfm_revenue_delta, 6),
        },
        "scope": "offline_policy_backtest_not_causal_promotional_incrementality",
        "causal_validity": "requires_online_randomized_experiment",
    }


# ===== Visualization =====
def _write_plot(evaluation: dict, output_path: Path) -> None:
    policies = evaluation["policy_comparison"]
    policy_names = [row["policy"] for row in policies]
    revenue_per_user = [row["actual_revenue_per_targeted_user"] for row in policies]
    purchase_rates = [row["purchase_rate"] for row in policies]

    fig, axis_left = plt.subplots(figsize=(8, 5))
    axis_left.bar(policy_names, revenue_per_user, color=["tab:blue", "tab:gray", "tab:orange"])
    axis_left.set_ylabel("Actual revenue per targeted user")
    axis_left.set_xlabel("Policy")
    axis_left.set_title("Offline Policy Backtest (Holdout Window)")
    axis_left.tick_params(axis="x", rotation=10)

    axis_right = axis_left.twinx()
    axis_right.plot(policy_names, purchase_rates, marker="o", color="tab:green")
    axis_right.set_ylabel("Actual purchase rate")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote evaluation plot: {output_path}")


def _write_model_diagnostics_plot(rows: list[dict], output_path: Path) -> None:
    labels = [int(float(row["label_purchase_30d"])) for row in rows]
    scores = [float(row["propensity_score"]) for row in rows]

    roc_fpr, roc_tpr, _ = roc_curve(labels, scores)
    pr_precision, pr_recall, _ = precision_recall_curve(labels, scores)

    scored_pairs = sorted(zip(scores, labels, strict=True), key=lambda row: row[0], reverse=True)
    decile_count = 10
    bin_size = max(1, len(scored_pairs) // decile_count)
    decile_lifts = []
    base_rate = sum(labels) / len(labels) if labels else 0.0
    for idx in range(decile_count):
        start = idx * bin_size
        end = len(scored_pairs) if idx == decile_count - 1 else min(len(scored_pairs), (idx + 1) * bin_size)
        if start >= len(scored_pairs):
            decile_lifts.append(0.0)
            continue
        segment = scored_pairs[start:end]
        segment_rate = sum(label for _, label in segment) / len(segment)
        decile_lifts.append(segment_rate / base_rate if base_rate > 0 else 0.0)

    bin_count = 10
    calibration_x = []
    calibration_y = []
    for idx in range(bin_count):
        lower = idx / bin_count
        upper = (idx + 1) / bin_count
        if idx == bin_count - 1:
            in_bin = [row_idx for row_idx, value in enumerate(scores) if lower <= value <= upper]
        else:
            in_bin = [row_idx for row_idx, value in enumerate(scores) if lower <= value < upper]
        if not in_bin:
            continue
        calibration_x.append(sum(scores[row_idx] for row_idx in in_bin) / len(in_bin))
        calibration_y.append(sum(labels[row_idx] for row_idx in in_bin) / len(in_bin))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(roc_fpr, roc_tpr, color="tab:blue", linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")

    axes[0, 1].plot(pr_recall, pr_precision, color="tab:green", linewidth=2)
    axes[0, 1].set_title("PR Curve")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")

    axes[1, 0].plot(calibration_x, calibration_y, marker="o", color="tab:orange")
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1, 0].set_title("Calibration Curve")
    axes[1, 0].set_xlabel("Predicted probability")
    axes[1, 0].set_ylabel("Observed purchase rate")

    axes[1, 1].plot(range(1, decile_count + 1), decile_lifts, marker="o", color="tab:purple")
    axes[1, 1].axhline(1.0, linestyle="--", color="gray")
    axes[1, 1].set_title("Decile Lift")
    axes[1, 1].set_xlabel("Score decile (1=highest)")
    axes[1, 1].set_ylabel("Lift vs base rate")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote diagnostics plot: {output_path}")


# ===== Entry Point =====
def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Evaluate offline policy backtest from validation predictions.")
    parser.add_argument(
        "--scores-csv",
        default="artifacts/purchase_propensity/validation_predictions.csv",
        help="Path to validation predictions CSV emitted by training",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/purchase_propensity/evaluation.json",
        help="Path to compact evaluation output JSON",
    )
    parser.add_argument(
        "--output-plot",
        default="artifacts/purchase_propensity/evaluation_policy_comparison.png",
        help="Path to evaluation plot",
    )
    parser.add_argument(
        "--output-diagnostics-plot",
        default="artifacts/purchase_propensity/evaluation_model_diagnostics.png",
        help="Path to model diagnostics plot (ROC, PR, calibration, lift)",
    )
    args = parser.parse_args()

    # ===== Input Checks =====
    scores_path = Path(args.scores_csv)
    if not scores_path.exists():
        raise FileNotFoundError(f"Validation predictions CSV not found: {scores_path}")

    # ===== Evaluate + Write Outputs =====
    rows = _load_rows(scores_path)
    evaluation = build_policy_backtest(rows)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote evaluation: {output_path}")
    _write_plot(evaluation, Path(args.output_plot))
    _write_model_diagnostics_plot(rows, Path(args.output_diagnostics_plot))


if __name__ == "__main__":
    main()
