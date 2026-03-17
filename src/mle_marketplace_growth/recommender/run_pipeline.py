"""Run recommender pipeline end-to-end from one command.

Workflow Steps:
1) Load runtime config and resolve artifact paths.
2) Train candidate models, evaluate them offline, and select one by validation Recall@K.
3) Generate Top-K predictions with the selected model.
4) Validate required artifacts and metric contracts.
5) Write a short interpretation report for reviewers.
"""

from __future__ import annotations

import argparse

from mle_marketplace_growth.recommender.helpers.config import artifact_paths, load_recommender_runtime_config
from mle_marketplace_growth.recommender.predict import run_predict
from mle_marketplace_growth.recommender.train_and_select import run_train_and_select
from mle_marketplace_growth.recommender.validate_outputs import run_validate_outputs


def main() -> None:
    """What: Run recommender train, predict, validate, and interpretation in sequence.
    Why: Provides one deterministic end-to-end command for offline retrieval workflow.
    """
    # ===== CLI Args =====
    parser = argparse.ArgumentParser(description="Run recommender pipeline end-to-end.")
    parser.add_argument("--config", required=True, help="YAML config file")
    args = parser.parse_args()

    # ===== Load Config =====
    runtime = load_recommender_runtime_config(args.config)
    artifacts_dir = runtime.artifacts_dir
    paths = artifact_paths(runtime)

    # ===== Train + Select Model =====
    # Train candidate models and select by validation Recall@20.
    run_train_and_select(config_path=args.config)

    # ===== Predict Top-K =====
    # Generate serving-style Top-K predictions for all users.
    run_predict(config_path=args.config)

    # ===== Validate + Write Outputs =====
    # Validate outputs and write interpretation summary.
    _, interpretation_path = run_validate_outputs(
        artifacts_dir=artifacts_dir,
        output_json=paths.output_validation_summary,
        output_md=paths.output_interpretation,
    )
    print(f"Wrote output validation summary: {paths.output_validation_summary}")
    print(f"Wrote output interpretation: {interpretation_path}")


if __name__ == "__main__":
    main()
