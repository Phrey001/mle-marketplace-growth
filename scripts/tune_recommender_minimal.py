"""Run a minimal fixed-grid two-tower recommender tuning sweep.

Workflow Steps:
1) Load the base recommender YAML config.
2) Run one baseline trial using the YAML values exactly as provided.
3) Run additional trials by applying fixed two-tower sweep overrides on top of those YAML defaults.
3) Write one summary plus one compact folder per trial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mle_marketplace_growth.helpers import cfg_required
from mle_marketplace_growth.recommender.train import run_train

# Fixed two-tower override sets applied on top of the base YAML defaults.
TWO_TOWER_SWEEP_OVERRIDES = [
    {"temperature": 0.5},
    {"temperature": 1.0},
    {"negative_samples": 16},
    {"batch_size": 2048},
    {"early_stop_tolerance": 5e-4},
    {"temperature": 0.5, "negative_samples": 16},
    {"temperature": 0.5, "batch_size": 2048},
    {"temperature": 0.5, "negative_samples": 16, "batch_size": 2048, "early_stop_tolerance": 5e-4},
]


def _recall_at_20(metrics_path: Path, model_name: str) -> float:
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))["rows"]
    row = next((result for result in rows if result["model_name"] == model_name), None)
    return float(row["metrics"].get("Recall@20", 0.0)) if row else 0.0


def _trial_result(output_dir: Path, trial_name: str, config: dict[str, str | int | float]) -> dict:
    train_metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
    selected_model = str(train_metrics["selected_model_name"])
    validation_metrics_path = output_dir / "validation_retrieval_metrics.json"
    test_metrics_path = output_dir / "test_retrieval_metrics.json"
    return {
        "trial_name": trial_name,
        "config": config,
        "selected_model_name": selected_model,
        "validation_recall_at_20": _recall_at_20(validation_metrics_path, selected_model),
        "test_recall_at_20": _recall_at_20(test_metrics_path, selected_model),
        "two_tower_validation_recall_at_20": _recall_at_20(validation_metrics_path, "two_tower"),
        "two_tower_test_recall_at_20": _recall_at_20(test_metrics_path, "two_tower"),
    }


def _remove_tuning_bloat(output_dir: Path) -> None:
    """What: Remove per-trial artifacts not needed for tuning review.
    Why: Keeps the tuning output contract focused on config plus validation/test evidence.
    """
    for artifact_name in ["model_bundle.pkl", "train_metrics.json"]:
        artifact_path = output_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal fixed-grid two-tower recommender tuning sweep.")
    parser.add_argument("--config", required=True, help="Base recommender YAML config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_payload, dict):
        raise ValueError("Config file must contain a key-value object")

    output_root = Path("artifacts") / "recommender" / "tuning"
    output_root.mkdir(parents=True, exist_ok=True)
    default_cfg = {
        "embedding_dim": int(cfg_required(config_payload, "embedding_dim")),
        "epochs": int(cfg_required(config_payload, "epochs")),
        "learning_rate": float(cfg_required(config_payload, "learning_rate")),
        "negative_samples": int(cfg_required(config_payload, "negative_samples")),
        "batch_size": int(cfg_required(config_payload, "batch_size")),
        "l2_reg": float(cfg_required(config_payload, "l2_reg")),
        "max_grad_norm": float(cfg_required(config_payload, "max_grad_norm")),
        "early_stop_rounds": int(cfg_required(config_payload, "early_stop_rounds")),
        "early_stop_k": int(cfg_required(config_payload, "early_stop_k")),
        "early_stop_tolerance": float(cfg_required(config_payload, "early_stop_tolerance")),
        "temperature": float(cfg_required(config_payload, "temperature")),
        "mf_components": int(cfg_required(config_payload, "mf_components")),
        "mf_n_iter": int(cfg_required(config_payload, "mf_n_iter")),
        "mf_weighting": str(cfg_required(config_payload, "mf_weighting")),
    }
    # Each sweep trial starts from the YAML defaults, then replaces only the listed two-tower knobs below.
    small_grid = [{**default_cfg, **overrides} for overrides in TWO_TOWER_SWEEP_OVERRIDES]

    results: list[dict] = []
    # `trial_default` uses the YAML values exactly. `trial_i` applies one fixed override set on top of those defaults.
    trial_specs = [("trial_default", default_cfg)] + [(f"trial_{i}", cfg) for i, cfg in enumerate(small_grid, start=1)]
    for trial_name, trial_overrides in trial_specs:
        trial_dir = output_root / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_config = {
            **config_payload,
            **trial_overrides,
        }
        trial_config_path = trial_dir / "trial_config.yaml"
        trial_config_path.write_text(yaml.safe_dump(trial_config, sort_keys=False), encoding="utf-8")
        run_train(config_path=str(trial_config_path), output_dir_override=trial_dir)
        results.append(_trial_result(trial_dir, trial_name, trial_overrides))
        _remove_tuning_bloat(trial_dir)

    best = max(results, key=lambda result: result["validation_recall_at_20"])
    best_two_tower = max(results, key=lambda result: result["two_tower_validation_recall_at_20"])
    summary = {
        "strategy": "fixed_small_grid_two_tower_local",
        "best_trial": best,
        "best_two_tower_trial": best_two_tower,
        "trials": results,
    }
    summary_path = output_root / "tuning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote tuning summary: {summary_path}")


if __name__ == "__main__":
    main()
