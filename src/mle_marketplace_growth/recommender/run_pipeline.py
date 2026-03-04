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


def _resolve_from_cfg(args: argparse.Namespace, cfg_get, name: str, default, cast=None):
    value = getattr(args, name)
    value = cfg_get(name, default) if value is None else value
    return cast(value) if cast else value


def main() -> None:
    # Parse CLI + config defaults.
    parser = argparse.ArgumentParser(description="Run recommender pipeline end-to-end.")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--output-root", default=None, help="Root containing prebuilt recommender gold feature-store datasets")
    parser.add_argument("--artifacts-dir", default=None, help="Recommender artifacts output directory")
    parser.add_argument("--split-version", default=None, help="Split strategy/version for user_item_splits")
    parser.add_argument("--recommender-min-event-date", default=None, help="Lower bound date (YYYY-MM-DD) for recommender interaction events.")
    parser.add_argument("--recommender-max-event-date", default=None, help="Upper bound date (YYYY-MM-DD) for recommender interaction events.")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Two-tower embedding dimension")
    parser.add_argument("--epochs", type=int, default=None, help="Two-tower training epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="Two-tower optimizer learning rate")
    parser.add_argument("--negative-samples", type=int, default=None, help="Negative samples per positive interaction")
    parser.add_argument("--batch-size", type=int, default=None, help="Two-tower positive-pair batch size")
    parser.add_argument("--l2-reg", type=float, default=None, help="L2 regularization strength")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Two-tower gradient clipping norm (0 disables)")
    parser.add_argument("--early-stop-rounds", type=int, default=None, help="Two-tower early stop rounds (0 disables)")
    parser.add_argument("--early-stop-metric", choices=["loss", "val_recall_at_k"], default=None, help="Two-tower early-stop metric")
    parser.add_argument("--early-stop-k", type=int, default=None, help="K used by validation Recall@K early stopping")
    parser.add_argument("--early-stop-tolerance", type=float, default=None, help="Minimum metric improvement for two-tower progress")
    parser.add_argument("--temperature", type=float, default=None, help="Softmax temperature for two-tower logits (>0)")
    parser.add_argument("--normalize-embeddings", type=int, choices=[0, 1], default=None, help="L2-normalize two-tower embeddings during scoring (1=yes)")
    parser.add_argument("--tower-hidden-dim", type=int, default=None, help="Two-tower MLP hidden dimension (0 disables MLP tower)")
    parser.add_argument("--tower-dropout", type=float, default=None, help="Two-tower MLP dropout rate")
    parser.add_argument("--device", choices=["auto"], default=None, help="Device mode for two-tower training (auto=cuda if available else cpu)")
    parser.add_argument("--mf-components", type=int, default=None, help="Latent factors for MF baseline")
    parser.add_argument("--mf-n-iter", type=int, default=None, help="Iteration budget for MF solver")
    parser.add_argument("--mf-weighting", choices=["binary", "tfidf"], default=None, help="Input weighting mode for MF")
    parser.add_argument("--mf-algorithm", choices=["randomized", "arpack"], default=None, help="MF SVD solver")
    parser.add_argument("--mf-tol", type=float, default=None, help="MF error tolerance (used by arpack solver)")
    parser.add_argument("--popularity-transform", choices=["linear", "log1p"], default=None, help="Score transform for popularity baseline")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K recommendations per user")
    parser.add_argument("--top-ks", default=None, help="Comma-separated K values for metrics")
    args = parser.parse_args()
    cfg = load_yaml_defaults(args.config, "Engine config").get
    for name, default, cast in [
        ("output_root", "data", None),
        ("artifacts_dir", "artifacts/recommender", None),
        ("split_version", "time_rank_v1", None),
        ("recommender_min_event_date", "2010-12-01", None),
        ("recommender_max_event_date", "2011-11-30", None),
        ("embedding_dim", 64, int),
        ("epochs", 10, int),
        ("learning_rate", 0.01, float),
        ("negative_samples", 8, int),
        ("batch_size", 4096, int),
        ("l2_reg", 1e-4, float),
        ("max_grad_norm", 1.0, float),
        ("early_stop_rounds", 3, int),
        ("early_stop_metric", "val_recall_at_k", None),
        ("early_stop_k", 20, int),
        ("early_stop_tolerance", 1e-4, float),
        ("temperature", 1.0, float),
        ("normalize_embeddings", 1, int),
        ("tower_hidden_dim", 0, int),
        ("tower_dropout", 0.0, float),
        ("device", "auto", None),
        ("mf_components", 64, int),
        ("mf_n_iter", 15, int),
        ("mf_weighting", "tfidf", None),
        ("mf_algorithm", "randomized", None),
        ("mf_tol", 0.0, float),
        ("popularity_transform", "log1p", None),
        ("top_k", 20, int),
        ("top_ks", "10,20", None),
    ]:
        setattr(args, name, _resolve_from_cfg(args, cfg, name, default, cast))

    # Resolve prebuilt gold inputs.
    output_root = Path(args.output_root)
    artifacts_dir = Path(args.artifacts_dir)

    split_csv = output_root / "gold" / "feature_store" / "recommender" / "user_item_splits" / "user_item_splits.csv"
    user_index_csv = output_root / "gold" / "feature_store" / "recommender" / "user_index" / "user_index.csv"
    item_index_csv = output_root / "gold" / "feature_store" / "recommender" / "item_index" / "item_index.csv"
    for required_path in [split_csv, user_index_csv, item_index_csv]:
        if not required_path.exists():
            raise FileNotFoundError(
                "Missing prebuilt recommender gold dataset. "
                "Build it first with `mle_marketplace_growth.feature_store.build_gold_recommender` "
                f"(expected path: {required_path})."
            )

    # Train candidate models and select by validation Recall@20.
    _run_module(
        "mle_marketplace_growth.recommender.train",
        "--splits-csv",
        split_csv,
        "--user-index-csv",
        user_index_csv,
        "--item-index-csv",
        item_index_csv,
        "--output-dir",
        artifacts_dir,
        "--embedding-dim",
        args.embedding_dim,
        "--epochs",
        args.epochs,
        "--learning-rate",
        args.learning_rate,
        "--negative-samples",
        args.negative_samples,
        "--batch-size",
        args.batch_size,
        "--l2-reg",
        args.l2_reg,
        "--max-grad-norm",
        args.max_grad_norm,
        "--early-stop-rounds",
        args.early_stop_rounds,
        "--early-stop-metric",
        args.early_stop_metric,
        "--early-stop-k",
        args.early_stop_k,
        "--early-stop-tolerance",
        args.early_stop_tolerance,
        "--temperature",
        args.temperature,
        "--normalize-embeddings",
        args.normalize_embeddings,
        "--tower-hidden-dim",
        args.tower_hidden_dim,
        "--tower-dropout",
        args.tower_dropout,
        "--device",
        args.device,
        "--mf-components",
        args.mf_components,
        "--mf-n-iter",
        args.mf_n_iter,
        "--mf-weighting",
        args.mf_weighting,
        "--mf-algorithm",
        args.mf_algorithm,
        "--mf-tol",
        args.mf_tol,
        "--popularity-transform",
        args.popularity_transform,
        "--top-ks",
        args.top_ks,
    )

    # Generate serving-style Top-K predictions for all users.
    _run_module(
        "mle_marketplace_growth.recommender.predict",
        "--model-bundle",
        artifacts_dir / "model_bundle.pkl",
        "--output-csv",
        artifacts_dir / "topk_recommendations.csv",
        "--top-k",
        args.top_k,
    )

    # Validate outputs and write interpretation summary.
    summary_path = artifacts_dir / "output_validation_summary.json"
    passed, summary = run_validation(artifacts_dir=artifacts_dir, output_json=summary_path)
    if not passed: raise ValueError(f"Automated artifact validation failed: {[row for row in summary['checks'] if not row['passed']]}")
    print(f"Wrote output validation summary: {summary_path}")
    interpretation_path = write_interpretation(artifacts_dir=artifacts_dir)
    print(f"Wrote output interpretation: {interpretation_path}")


if __name__ == "__main__":
    main()
