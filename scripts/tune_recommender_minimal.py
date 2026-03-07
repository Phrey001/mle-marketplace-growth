"""Run a minimal fixed-grid recommender hyperparameter sweep.

Execution order:
1) run `trial_default` from the provided recommender YAML config
2) run compact two-tower-focused grid variants for comparison
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _run_train(
    splits_path: Path,
    user_index_path: Path,
    item_index_path: Path,
    output_dir: Path,
    top_ks: str,
    config: dict[str, str | int | float],
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "mle_marketplace_growth.recommender.train",
        "--splits-path",
        str(splits_path),
        "--user-index-path",
        str(user_index_path),
        "--item-index-path",
        str(item_index_path),
        "--output-dir",
        str(output_dir),
        "--top-ks",
        top_ks,
    ]
    for key, value in config.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _recall_at_20(metrics_path: Path, model_name: str) -> float:
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))["rows"]
    row = next((r for r in rows if r["model_name"] == model_name), None)
    return float(row["metrics"].get("Recall@20", 0.0)) if row else 0.0


def _trial_result(output_dir: Path, trial_name: str, config: dict[str, str | int | float]) -> dict:
    train_metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
    selected_model = str(train_metrics["selected_model_name"])
    val_path, test_path = output_dir / "validation_retrieval_metrics.json", output_dir / "test_retrieval_metrics.json"
    return {
        "trial_name": trial_name,
        "config": config,
        "selected_model_name": selected_model,
        "validation_recall_at_20": _recall_at_20(val_path, selected_model),
        "test_recall_at_20": _recall_at_20(test_path, selected_model),
        "two_tower_validation_recall_at_20": _recall_at_20(val_path, "two_tower"),
        "two_tower_test_recall_at_20": _recall_at_20(test_path, "two_tower"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal recommender hyperparameter tuning sweep.")
    parser.add_argument("--config", default="configs/recommender/default.yaml", help="Base recommender YAML config")
    parser.add_argument("--splits-path", default="data/gold/feature_store/recommender/user_item_splits/user_item_splits.parquet")
    parser.add_argument("--user-index-path", default="data/gold/feature_store/recommender/user_index/user_index.parquet")
    parser.add_argument("--item-index-path", default="data/gold/feature_store/recommender/item_index/item_index.parquet")
    parser.add_argument("--output-root", default="artifacts/recommender/tuning")
    parser.add_argument("--top-ks", default="10,20")
    args = parser.parse_args()

    splits_path = Path(args.splits_path)
    user_index_path = Path(args.user_index_path)
    item_index_path = Path(args.item_index_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_payload, dict):
        raise ValueError("Config file must contain a key-value object")

    default_cfg = {
        "embedding_dim": int(config_payload.get("embedding_dim", 64)),
        "epochs": int(config_payload.get("epochs", 12)),
        "learning_rate": float(config_payload.get("learning_rate", 0.003)),
        "negative_samples": int(config_payload.get("negative_samples", 8)),
        "batch_size": int(config_payload.get("batch_size", 4096)),
        "l2_reg": float(config_payload.get("l2_reg", 1e-4)),
        "max_grad_norm": float(config_payload.get("max_grad_norm", 1.0)),
        "early_stop_rounds": int(config_payload.get("early_stop_rounds", 4)),
        "early_stop_k": int(config_payload.get("early_stop_k", 20)),
        "early_stop_tolerance": float(config_payload.get("early_stop_tolerance", 1e-4)),
        "temperature": float(config_payload.get("temperature", 1.0)),
        "tower_hidden_dim": int(config_payload.get("tower_hidden_dim", 0)),
        "tower_dropout": float(config_payload.get("tower_dropout", 0.0)),
        "mf_components": int(config_payload.get("mf_components", 64)),
        "mf_n_iter": int(config_payload.get("mf_n_iter", 15)),
        "mf_weighting": str(config_payload.get("mf_weighting", "tfidf")),
    }
    # Compact local sweep around current defaults (keeps runtime practical).
    small_grid = [
        {**default_cfg, "temperature": 0.5},
        {**default_cfg, "temperature": 1.0},
        {**default_cfg, "negative_samples": 16},
        {**default_cfg, "batch_size": 2048},
        {**default_cfg, "early_stop_tolerance": 5e-4},
        {**default_cfg, "temperature": 0.5, "negative_samples": 16},
        {**default_cfg, "temperature": 0.5, "batch_size": 2048},
        {**default_cfg, "temperature": 0.5, "negative_samples": 16, "batch_size": 2048, "early_stop_tolerance": 5e-4},
    ]

    results: list[dict] = []
    # Baseline trial uses current YAML defaults so tuning is comparable to main pipeline settings.
    trial_dir = output_root / "trial_default"
    _run_train(splits_path, user_index_path, item_index_path, trial_dir, args.top_ks, default_cfg)
    results.append(_trial_result(trial_dir, "trial_default", default_cfg))
    for i, cfg in enumerate(small_grid, start=1):
        tdir = output_root / f"trial_{i}"
        _run_train(splits_path, user_index_path, item_index_path, tdir, args.top_ks, cfg)
        results.append(_trial_result(tdir, f"trial_{i}", cfg))

    best = max(results, key=lambda r: r["validation_recall_at_20"])
    best_two_tower = max(results, key=lambda r: r["two_tower_validation_recall_at_20"])
    summary = {
        "strategy": "fixed_small_grid_two_tower_local",
        "best_trial": best,
        "best_two_tower_trial": best_two_tower,
        "trials": results,
    }
    out = output_root / "tuning_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote tuning summary: {out}")


if __name__ == "__main__":
    main()
