"""Generate policy comparison chart across initial and retrain cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_policy_values(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_policy = {row["policy"]: row for row in payload.get("policy_comparison", [])}
    return {
        "ml_top_expected_value": float(by_policy["ml_top_expected_value"]["actual_revenue_per_targeted_user"]),
        "random_baseline": float(by_policy["random_baseline"]["actual_revenue_per_targeted_user"]),
        "rfm_heuristic": float(by_policy["rfm_heuristic"]["actual_revenue_per_targeted_user"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create policy comparison chart for report assets.")
    parser.add_argument(
        "--initial-json",
        default="artifacts/purchase_propensity/cycle_initial/offline_eval/offline_policy_budget_test.json",
        help="Cycle-initial budget policy test JSON",
    )
    parser.add_argument(
        "--retrain-json",
        default="artifacts/purchase_propensity/cycle_retrain/offline_eval/offline_policy_budget_test.json",
        help="Cycle-retrain budget policy test JSON",
    )
    parser.add_argument(
        "--output-png",
        default="artifacts/purchase_propensity/report_assets/policy_comparison_cycles.png",
        help="Output chart PNG path",
    )
    args = parser.parse_args()

    initial_values = _load_policy_values(Path(args.initial_json))
    retrain_values = _load_policy_values(Path(args.retrain_json))
    values_by_policy = {
        "ml_top_expected_value": [initial_values["ml_top_expected_value"], retrain_values["ml_top_expected_value"]],
        "random_baseline": [initial_values["random_baseline"], retrain_values["random_baseline"]],
        "rfm_heuristic": [initial_values["rfm_heuristic"], retrain_values["rfm_heuristic"]],
    }
    labels = {"ml_top_expected_value": "ML EV", "random_baseline": "Random", "rfm_heuristic": "RFM"}
    colors = {"ml_top_expected_value": "#1f77b4", "random_baseline": "#ff7f0e", "rfm_heuristic": "#2ca02c"}

    x = [0, 1]
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for idx, policy in enumerate(["ml_top_expected_value", "random_baseline", "rfm_heuristic"]):
        offsets = [point + (idx - 1) * width for point in x]
        bars = ax.bar(offsets, values_by_policy[policy], width=width, label=labels[policy], color=colors[policy])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["Initial", "Retrain"])
    ax.set_ylabel("Actual Revenue per Targeted User")
    ax.set_title("Policy Comparison by Cycle (Budget-Constrained Top-K)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
