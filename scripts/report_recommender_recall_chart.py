"""Generate the recommender Recall@20 comparison chart from the canonical run artifacts.

Workflow Steps:
1) Load the recommender runtime config to find the canonical artifact folder.
2) Read validation/test retrieval metrics for the selected as-of date.
3) Write one report chart PNG under `artifacts/recommender/report_assets/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from mle_marketplace_growth.recommender.helpers.config import load_recommender_runtime_config


def _load_recall_at_20(path: Path) -> dict[str, float]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows", [])
    by_model = {row["model_name"]: row for row in rows}
    return {
        "popularity": float(by_model["popularity"]["metrics"]["Recall@20"]),
        "mf": float(by_model["mf"]["metrics"]["Recall@20"]),
        "two_tower": float(by_model["two_tower"]["metrics"]["Recall@20"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the recommender Recall@20 report chart.")
    parser.add_argument("--config", required=True, help="Recommender YAML config")
    args = parser.parse_args()

    runtime = load_recommender_runtime_config(args.config)
    validation = _load_recall_at_20(runtime.artifacts_dir / "validation_retrieval_metrics.json")
    test = _load_recall_at_20(runtime.artifacts_dir / "test_retrieval_metrics.json")
    values_by_model = {
        "popularity": [validation["popularity"], test["popularity"]],
        "mf": [validation["mf"], test["mf"]],
        "two_tower": [validation["two_tower"], test["two_tower"]],
    }
    labels = {"popularity": "Popularity", "mf": "MF", "two_tower": "Two-Tower"}
    colors = {"popularity": "#4e79a7", "mf": "#2ca02c", "two_tower": "#f28e2b"}

    x = [0, 1]
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for idx, model_name in enumerate(["popularity", "mf", "two_tower"]):
        offsets = [point + (idx - 1) * width for point in x]
        bars = ax.bar(offsets, values_by_model[model_name], width=width, label=labels[model_name], color=colors[model_name])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["Validation", "Test"])
    ax.set_ylabel("Recall@20")
    ax.set_title("Recommender Model Comparison (Recall@20)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = Path("artifacts") / "recommender" / "report_assets" / "model_recall_at20_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
