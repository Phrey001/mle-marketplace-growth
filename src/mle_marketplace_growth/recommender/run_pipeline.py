"""Run recommender pipeline end-to-end from one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from mle_marketplace_growth.recommender.validate_outputs import run_validation, write_interpretation

def _run_module(module: str, *args: object) -> None:
    command = [sys.executable, "-m", module, *map(str, args)]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def _load_yaml_defaults(path_value: str | None, label: str) -> dict:
    if not path_value:
        return {}
    config_path = Path(path_value)
    if not config_path.exists():
        raise FileNotFoundError(f"{label} file not found: {config_path}")
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"{label} file must use .yaml or .yml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} file must contain a key-value object")
    return payload


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional YAML config file")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    engine_defaults = _load_yaml_defaults(pre_args.config, "Engine config")
    cfg = engine_defaults.get

    parser = argparse.ArgumentParser(description="Run recommender pipeline end-to-end.")
    parser.add_argument("--config", default=pre_args.config, help="Optional YAML config file")
    parser.add_argument("--input-csv", default=cfg("input_csv", "data/bronze/online_retail_ii/raw.csv"), help="Raw source CSV")
    parser.add_argument("--output-root", default=cfg("output_root", "data"), help="Output root for feature-store materialization")
    parser.add_argument("--artifacts-dir", default=cfg("artifacts_dir", "artifacts/recommender"), help="Recommender artifacts output directory")
    parser.add_argument("--split-version", default=cfg("split_version", "time_rank_v1"), help="Split strategy/version for user_item_splits")
    parser.add_argument("--recommender-min-event-date", default=cfg("recommender_min_event_date", "2010-12-01"), help="Lower bound date (YYYY-MM-DD) for recommender interaction events.")
    parser.add_argument("--recommender-max-event-date", default=cfg("recommender_max_event_date", "2011-11-30"), help="Upper bound date (YYYY-MM-DD) for recommender interaction events.")
    parser.add_argument("--embedding-dim", type=int, default=int(cfg("embedding_dim", 64)), help="Two-tower embedding dimension")
    parser.add_argument("--epochs", type=int, default=int(cfg("epochs", 10)), help="Two-tower training epochs")
    parser.add_argument("--learning-rate", type=float, default=float(cfg("learning_rate", 0.01)), help="Two-tower optimizer learning rate")
    parser.add_argument("--negative-samples", type=int, default=int(cfg("negative_samples", 8)), help="Negative samples per positive interaction")
    parser.add_argument("--batch-size", type=int, default=int(cfg("batch_size", 4096)), help="Two-tower positive-pair batch size")
    parser.add_argument("--l2-reg", type=float, default=float(cfg("l2_reg", 1e-4)), help="L2 regularization strength")
    parser.add_argument("--max-grad-norm", type=float, default=float(cfg("max_grad_norm", 1.0)), help="Two-tower gradient clipping norm (0 disables)")
    parser.add_argument("--early-stop-rounds", type=int, default=int(cfg("early_stop_rounds", 3)), help="Two-tower early stop rounds (0 disables)")
    parser.add_argument("--early-stop-metric", choices=["loss", "val_recall_at_k"], default=cfg("early_stop_metric", "val_recall_at_k"), help="Two-tower early-stop metric")
    parser.add_argument("--early-stop-k", type=int, default=int(cfg("early_stop_k", 20)), help="K used by validation Recall@K early stopping")
    parser.add_argument("--early-stop-tolerance", type=float, default=float(cfg("early_stop_tolerance", 1e-4)), help="Minimum metric improvement for two-tower progress")
    parser.add_argument("--temperature", type=float, default=float(cfg("temperature", 1.0)), help="Softmax temperature for two-tower logits (>0)")
    parser.add_argument("--normalize-embeddings", type=int, choices=[0, 1], default=int(cfg("normalize_embeddings", 1)), help="L2-normalize two-tower embeddings during scoring (1=yes)")
    parser.add_argument("--tower-hidden-dim", type=int, default=int(cfg("tower_hidden_dim", 0)), help="Two-tower MLP hidden dimension (0 disables MLP tower)")
    parser.add_argument("--tower-dropout", type=float, default=float(cfg("tower_dropout", 0.0)), help="Two-tower MLP dropout rate")
    parser.add_argument("--device", choices=["auto"], default=cfg("device", "auto"), help="Device mode for two-tower training (auto=cuda if available else cpu)")
    parser.add_argument("--mf-components", type=int, default=int(cfg("mf_components", 64)), help="Latent factors for MF baseline")
    parser.add_argument("--mf-n-iter", type=int, default=int(cfg("mf_n_iter", 15)), help="Iteration budget for MF solver")
    parser.add_argument("--mf-weighting", choices=["binary", "tfidf"], default=cfg("mf_weighting", "tfidf"), help="Input weighting mode for MF")
    parser.add_argument("--mf-algorithm", choices=["randomized", "arpack"], default=cfg("mf_algorithm", "randomized"), help="MF SVD solver")
    parser.add_argument("--mf-tol", type=float, default=float(cfg("mf_tol", 0.0)), help="MF error tolerance (used by arpack solver)")
    parser.add_argument("--popularity-transform", choices=["linear", "log1p"], default=cfg("popularity_transform", "log1p"), help="Score transform for popularity baseline")
    parser.add_argument("--top-k", type=int, default=int(cfg("top_k", 20)), help="Top-K recommendations per user")
    parser.add_argument("--top-ks", default=cfg("top_ks", "10,20"), help="Comma-separated K values for metrics")
    args = parser.parse_args(remaining_argv)

    output_root = Path(args.output_root)
    artifacts_dir = Path(args.artifacts_dir)

    build_args: list[object] = [
        "--build-engines",
        "recommender",
        "--input-csv",
        args.input_csv,
        "--output-root",
        output_root,
        "--split-version",
        args.split_version,
        "--recommender-min-event-date",
        args.recommender_min_event_date,
        "--recommender-max-event-date",
        args.recommender_max_event_date,
    ]
    _run_module(
        "mle_marketplace_growth.feature_store.build",
        *build_args,
    )

    split_csv = output_root / "gold" / "feature_store" / "recommender" / "user_item_splits" / "user_item_splits.csv"
    user_index_csv = output_root / "gold" / "feature_store" / "recommender" / "user_index" / "user_index.csv"
    item_index_csv = output_root / "gold" / "feature_store" / "recommender" / "item_index" / "item_index.csv"
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

    _run_module(
        "mle_marketplace_growth.recommender.predict",
        "--model-bundle",
        artifacts_dir / "model_bundle.pkl",
        "--output-csv",
        artifacts_dir / "topk_recommendations.csv",
        "--top-k",
        args.top_k,
    )

    summary_path = artifacts_dir / "output_validation_summary.json"
    passed, summary = run_validation(artifacts_dir=artifacts_dir, output_json=summary_path)
    if not passed: raise ValueError(f"Automated artifact validation failed: {[row for row in summary['checks'] if not row['passed']]}")
    print(f"Wrote output validation summary: {summary_path}")
    interpretation_path = write_interpretation(artifacts_dir=artifacts_dir)
    print(f"Wrote output interpretation: {interpretation_path}")


if __name__ == "__main__":
    main()
