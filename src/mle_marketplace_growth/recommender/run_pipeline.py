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

def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional YAML config file")
    pre_args, remaining_argv = pre_parser.parse_known_args()
    if pre_args.config:
        config_path = Path(pre_args.config)
        if config_path.suffix.lower() not in {".yaml", ".yml"}: raise ValueError("Config file must use .yaml or .yml")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict): raise ValueError("Config file must contain a key-value object")
        defaults = payload
    else:
        defaults = {}
    cfg = defaults.get

    parser = argparse.ArgumentParser(description="Run recommender pipeline end-to-end.")
    parser.add_argument("--config", default=pre_args.config, help="Optional YAML config file")
    parser.add_argument("--input-csv", default=cfg("input_csv", "data/bronze/online_retail_ii/raw.csv"), help="Raw source CSV")
    parser.add_argument("--output-root", default=cfg("output_root", "data"), help="Output root for feature-store materialization")
    parser.add_argument("--artifacts-dir", default=cfg("artifacts_dir", "artifacts/recommender"), help="Recommender artifacts output directory")
    parser.add_argument("--split-version", default=cfg("split_version", "time_rank_v1"), help="Split strategy/version for user_item_splits")
    parser.add_argument("--recommender-min-event-date", default=cfg("recommender_min_event_date", "2010-12-01"), help="Lower bound date (YYYY-MM-DD) for recommender interaction events.")
    parser.add_argument("--recommender-max-event-date", default=cfg("recommender_max_event_date", "2011-11-30"), help="Upper bound date (YYYY-MM-DD) for recommender interaction events.")
    parser.add_argument("--top-k", type=int, default=int(cfg("top_k", 20)), help="Top-K recommendations per user")
    parser.add_argument("--top-ks", default=cfg("top_ks", "10,20"), help="Comma-separated K values for metrics")
    args = parser.parse_args(remaining_argv)

    output_root = Path(args.output_root)
    artifacts_dir = Path(args.artifacts_dir)

    _run_module(
        "mle_marketplace_growth.feature_store.build",
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
