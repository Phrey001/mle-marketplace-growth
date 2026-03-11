"""Run recommender pipeline end-to-end from one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mle_marketplace_growth.feature_store.build_helpers import load_yaml_defaults
from mle_marketplace_growth.recommender.validate_outputs import run_validation, write_interpretation

def _run_module(module: str, *args: object) -> None:
    command = [sys.executable, "-m", module, *map(str, args)]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Run recommender pipeline end-to-end.")
    parser.add_argument("--config", required=True, help="YAML config file")
    args = parser.parse_args()

    # ===== Load Config =====
    cfg = load_yaml_defaults(args.config, "Engine config").get
    output_root = Path(cfg("output_root", "data"))
    artifacts_dir = Path(cfg("artifacts_dir", "artifacts/recommender"))
    embedding_dim = int(cfg("embedding_dim", 64))
    epochs = int(cfg("epochs", 10))
    learning_rate = float(cfg("learning_rate", 0.01))
    negative_samples = int(cfg("negative_samples", 8))
    batch_size = int(cfg("batch_size", 4096))
    l2_reg = float(cfg("l2_reg", 1e-4))
    max_grad_norm = float(cfg("max_grad_norm", 1.0))
    early_stop_rounds = int(cfg("early_stop_rounds", 3))
    early_stop_k = int(cfg("early_stop_k", 20))
    early_stop_tolerance = float(cfg("early_stop_tolerance", 1e-4))
    temperature = float(cfg("temperature", 1.0))
    tower_hidden_dim = int(cfg("tower_hidden_dim", 0))
    tower_dropout = float(cfg("tower_dropout", 0.0))
    mf_components = int(cfg("mf_components", 64))
    mf_n_iter = int(cfg("mf_n_iter", 15))
    mf_weighting = cfg("mf_weighting", "tfidf")
    top_k = int(cfg("top_k", 20))
    top_ks = cfg("top_ks", "10,20")

    # ===== Validate Inputs (Prebuilt Gold Required) =====
    split_path = output_root / "gold" / "feature_store" / "recommender" / "user_item_splits" / "user_item_splits.parquet"
    user_index_path = output_root / "gold" / "feature_store" / "recommender" / "user_index" / "user_index.parquet"
    item_index_path = output_root / "gold" / "feature_store" / "recommender" / "item_index" / "item_index.parquet"
    for required_path in [split_path, user_index_path, item_index_path]:
        if not required_path.exists():
            raise FileNotFoundError(
                "Missing prebuilt recommender gold dataset. "
                "Build it first with `mle_marketplace_growth.feature_store.build_gold_recommender` "
                f"(expected path: {required_path})."
            )

    # ===== Train + Select Model =====
    # Train candidate models and select by validation Recall@20.
    _run_module(
        "mle_marketplace_growth.recommender.train",
        "--splits-path",
        split_path,
        "--user-index-path",
        user_index_path,
        "--item-index-path",
        item_index_path,
        "--output-dir",
        artifacts_dir,
        "--embedding-dim",
        embedding_dim,
        "--epochs",
        epochs,
        "--learning-rate",
        learning_rate,
        "--negative-samples",
        negative_samples,
        "--batch-size",
        batch_size,
        "--l2-reg",
        l2_reg,
        "--max-grad-norm",
        max_grad_norm,
        "--early-stop-rounds",
        early_stop_rounds,
        "--early-stop-k",
        early_stop_k,
        "--early-stop-tolerance",
        early_stop_tolerance,
        "--temperature",
        temperature,
        "--tower-hidden-dim",
        tower_hidden_dim,
        "--tower-dropout",
        tower_dropout,
        "--mf-components",
        mf_components,
        "--mf-n-iter",
        mf_n_iter,
        "--mf-weighting",
        mf_weighting,
        "--top-ks",
        top_ks,
    )

    # ===== Predict Top-K =====
    # Generate serving-style Top-K predictions for all users.
    _run_module(
        "mle_marketplace_growth.recommender.predict",
        "--model-bundle",
        artifacts_dir / "model_bundle.pkl",
        "--output-csv",
        artifacts_dir / "topk_recommendations.csv",
        "--top-k",
        top_k,
    )

    # ===== Validate + Interpret =====
    # Validate outputs and write interpretation summary.
    summary_path = artifacts_dir / "output_validation_summary.json"
    passed, summary = run_validation(artifacts_dir=artifacts_dir, output_json=summary_path)
    if not passed: raise ValueError(f"Automated artifact validation failed: {[row for row in summary['checks'] if not row['passed']]}")
    print(f"Wrote output validation summary: {summary_path}")
    interpretation_path = write_interpretation(artifacts_dir=artifacts_dir)
    print(f"Wrote output interpretation: {interpretation_path}")


if __name__ == "__main__":
    main()
