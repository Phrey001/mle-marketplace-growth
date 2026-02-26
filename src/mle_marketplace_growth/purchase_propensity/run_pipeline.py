"""Run the purchase propensity pipeline end-to-end from one command."""

import argparse
import calendar
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml

from mle_marketplace_growth.purchase_propensity.validate_outputs import run_validation, write_interpretation

ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}
ALLOWED_FEATURE_LOOKBACK_WINDOWS = {60, 90, 120}


# ===== Path + Date Helpers =====
def _add_month(current: date) -> date:
    year = current.year + (1 if current.month == 12 else 0)
    month = 1 if current.month == 12 else current.month + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _generate_snapshot_dates(start_date: date, end_date: date) -> list[str]:
    if end_date < start_date: raise ValueError("--train-end-date must be on or after --train-start-date")

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.isoformat())
        current = _add_month(current)
    return dates


def _merge_train_datasets(train_paths: list[Path], merged_output_path: Path) -> None:
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with merged_output_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = None
        for path in train_paths:
            if not path.exists(): raise FileNotFoundError(f"Training dataset not found: {path}")
            with path.open("r", encoding="utf-8", newline="") as in_file:
                reader = csv.DictReader(in_file)
                if not reader.fieldnames: raise ValueError(f"Missing header in training dataset: {path}")
                if writer is None: writer = csv.DictWriter(out_file, fieldnames=reader.fieldnames)
                if not wrote_header:
                    writer.writeheader()
                    wrote_header = True
                for row in reader:
                    writer.writerow(row)

def _run_module(module: str, *args: object) -> None:
    command = [sys.executable, "-m", module, *map(str, args)]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)

# ===== Entry Point =====
def main() -> None:
    # ===== Parse Config Defaults =====
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional YAML config file for pipeline arguments")
    pre_args, remaining_argv = pre_parser.parse_known_args()
    if pre_args.config:
        config_path = Path(pre_args.config)
        if not config_path.exists(): raise FileNotFoundError(f"Config file not found: {config_path}")
        if config_path.suffix.lower() not in {".yaml", ".yml"}: raise ValueError("Config file must use .yaml or .yml")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict): raise ValueError("Config file must contain a key-value object")
        config_defaults = payload
    else:
        config_defaults = {}
    cfg = config_defaults.get

    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Run purchase propensity pipeline end-to-end.")
    parser.add_argument("--config", default=pre_args.config, help="Optional YAML config file for pipeline arguments")

    parser.add_argument("--input-csv", default=cfg("input_csv", "data/bronze/online_retail_ii/raw.csv"), help="Path to raw source CSV")
    parser.add_argument("--output-root", default=cfg("output_root", "data"), help="Output root for feature-store build")
    parser.add_argument("--artifacts-dir", default=cfg("artifacts_dir", "artifacts/purchase_propensity"), help="Output directory for model/evaluation artifacts")
    parser.add_argument("--train-as-of-dates", default=cfg("train_as_of_dates", ""), help="Comma-separated snapshots used for model training (advanced override)")
    parser.add_argument("--train-start-date", default=cfg("train_start_date", None), help="Optional start date (YYYY-MM-DD) for generated training snapshots")
    parser.add_argument("--train-end-date", default=cfg("train_end_date", None), help="Optional end date (YYYY-MM-DD) for generated training snapshots")
    parser.add_argument("--train-frequency", choices=["monthly"], default=cfg("train_frequency", "monthly"), help="Frequency for generated training snapshots (monthly only)")
    parser.add_argument("--score-as-of-date", default=cfg("score_as_of_date", "2011-11-09"), help="Feature snapshot date used for scoring/evaluation")
    parser.add_argument("--prediction-window-days", type=int, default=int(cfg("prediction_window_days", 30)), help="Allowed values: 30, 60, 90 (strict demo execution path uses 30).")
    parser.add_argument("--feature-lookback-days", type=int, default=int(cfg("feature_lookback_days", 90)), help="Allowed values: 60, 90, 120 (strict demo execution path uses 90).")
    parser.add_argument("--window-selection-mode", choices=["sensitivity", "fixed"], default=cfg("window_selection_mode", "sensitivity"), help="`sensitivity`: select structural decisions from window_sensitivity output; `fixed`: use config values directly.")
    parser.add_argument("--force-propensity-model", choices=["logistic_regression", "xgboost"], default=cfg("force_propensity_model", None), help="Only used when --window-selection-mode=fixed.")
    parser.add_argument("--budget", type=float, default=cfg("budget", 5000.0), help="Budget for offline policy evaluation")
    parser.add_argument("--cost-per-user", type=float, default=cfg("cost_per_user", 5.0), help="Cost per targeted user for offline policy evaluation")
    args = parser.parse_args(remaining_argv)

    # ===== Validate Inputs =====
    input_csv = Path(args.input_csv)
    output_root = Path(args.output_root)
    artifacts_dir = Path(args.artifacts_dir)
    use_generated_dates = bool(args.train_start_date or args.train_end_date)
    if not use_generated_dates and not args.train_as_of_dates.strip(): args.train_as_of_dates = "2011-11-09"
    if use_generated_dates and args.train_as_of_dates.strip(): raise ValueError("Use either --train-as-of-dates or --train-start-date/--train-end-date, not both")
    if use_generated_dates:
        if not args.train_start_date or not args.train_end_date: raise ValueError("Both --train-start-date and --train-end-date are required together")
        start_date = date.fromisoformat(args.train_start_date)
        end_date = date.fromisoformat(args.train_end_date)
        train_as_of_dates = _generate_snapshot_dates(start_date, end_date)
    else:
        train_as_of_dates = [value.strip() for value in args.train_as_of_dates.split(",") if value.strip()]

    if not train_as_of_dates: raise ValueError("--train-as-of-dates must include at least one date")
    if len(train_as_of_dates) != 12: raise ValueError("Strict architecture split requires exactly 12 --train-as-of-dates (10 train / 1 val / 1 test)")
    if args.prediction_window_days not in ALLOWED_PREDICTION_WINDOWS: raise ValueError("--prediction-window-days must be one of: 30, 60, 90")
    if args.feature_lookback_days not in ALLOWED_FEATURE_LOOKBACK_WINDOWS: raise ValueError("--feature-lookback-days must be one of: 60, 90, 120")
    print(f"Window profile: prediction={args.prediction_window_days}d, feature_lookback={args.feature_lookback_days}d")

    # ===== Build Feature-Store Snapshots =====
    snapshots_to_build = set(train_as_of_dates)
    snapshots_to_build.add(args.score_as_of_date)
    for as_of_date in sorted(snapshots_to_build):
        _run_module(
            "mle_marketplace_growth.feature_store.build",
            "--build-engines",
            "purchase_propensity",
            "--input-csv",
            input_csv,
            "--output-root",
            output_root,
            "--as-of-date",
            as_of_date,
        )

    # ===== Build Training Input =====
    train_paths = [
        output_root
        / "gold"
        / "feature_store"
        / "purchase_propensity"
        / "propensity_train_dataset"
        / f"as_of_date={as_of_date}"
        / "propensity_train_dataset.csv"
        for as_of_date in train_as_of_dates
    ]
    if len(train_paths) == 1:
        train_input_csv = train_paths[0]
    else:
        train_input_csv = artifacts_dir / "_tmp" / "propensity_train_dataset_merged.csv"
        _merge_train_datasets(train_paths, train_input_csv)

    # ===== Structural Decision: Sensitivity Freeze or Fixed Config =====
    expect_window_sensitivity = args.window_selection_mode == "sensitivity"
    if expect_window_sensitivity:
        _run_module(
            "mle_marketplace_growth.purchase_propensity.window_sensitivity",
            "--input-csv",
            train_input_csv,
            "--events-csv",
            output_root / "silver" / "transactions_line_items" / "transactions_line_items.csv",
            "--output-json",
            artifacts_dir / "window_sensitivity.json",
            "--output-plot",
            artifacts_dir / "window_validation_dashboard.png",
        )
        window_sensitivity_output = json.loads((artifacts_dir / "window_sensitivity.json").read_text(encoding="utf-8"))
        freeze_decision = window_sensitivity_output["freeze_decision"]
        frozen_prediction_window_days = int(freeze_decision["selected_prediction_window_days"])
        frozen_feature_lookback_days = int(freeze_decision["selected_feature_lookback_days"])
        frozen_propensity_model = freeze_decision["selected_propensity_model_name"]
        print(
            "Frozen decision from sensitivity:",
            f"prediction_window={frozen_prediction_window_days}d,",
            f"feature_lookback={frozen_feature_lookback_days}d,",
            f"propensity_model={frozen_propensity_model}",
        )
    else:
        frozen_prediction_window_days = int(args.prediction_window_days)
        frozen_feature_lookback_days = int(args.feature_lookback_days)
        frozen_propensity_model = args.force_propensity_model
        print(
            "Frozen decision from fixed config:",
            f"prediction_window={frozen_prediction_window_days}d,",
            f"feature_lookback={frozen_feature_lookback_days}d,",
            f"propensity_model={frozen_propensity_model}",
        )

    # ===== Train + Predict + Evaluate =====
    train_args: list[str | Path | int] = [
        "--input-csv",
        train_input_csv,
        "--output-dir",
        artifacts_dir,
        "--prediction-window-days",
        frozen_prediction_window_days,
        "--feature-lookback-days",
        frozen_feature_lookback_days,
    ]
    if frozen_propensity_model: train_args.extend(["--force-propensity-model", frozen_propensity_model])
    _run_module("mle_marketplace_growth.purchase_propensity.train", *train_args)

    user_features_path = (
        output_root
        / "gold"
        / "feature_store"
        / "purchase_propensity"
        / "user_features_asof"
        / f"as_of_date={args.score_as_of_date}"
        / "user_features_asof.csv"
    )
    model_path = artifacts_dir / "propensity_model.pkl"
    prediction_scores_path = artifacts_dir / "prediction_scores.csv"
    validation_predictions_path = artifacts_dir / "validation_predictions.csv"
    test_predictions_path = artifacts_dir / "test_predictions.csv"
    budget_eval_validation_json_path = artifacts_dir / "offline_policy_budget_validation.json"
    budget_eval_test_json_path = artifacts_dir / "offline_policy_budget_test.json"

    _run_module(
        "mle_marketplace_growth.purchase_propensity.predict",
        "--input-csv",
        user_features_path,
        "--model-path",
        model_path,
        "--output-csv",
        prediction_scores_path,
    )
    policy_eval_args = [
        "--budget",
        args.budget,
        "--cost-per-user",
        args.cost_per_user,
        "--purchase-label-col",
        f"label_purchase_{frozen_prediction_window_days}d",
        "--revenue-label-col",
        f"label_net_revenue_{frozen_prediction_window_days}d",
    ]
    _run_module(
        "mle_marketplace_growth.purchase_propensity.policy_budget_evaluation",
        "--scores-csv",
        validation_predictions_path,
        "--output-json",
        budget_eval_validation_json_path,
        *policy_eval_args,
    )
    _run_module(
        "mle_marketplace_growth.purchase_propensity.policy_budget_evaluation",
        "--scores-csv",
        test_predictions_path,
        "--output-json",
        budget_eval_test_json_path,
        *policy_eval_args,
    )
    # ===== Validation + Interpretation =====
    summary_path = artifacts_dir / "output_validation_summary.json"
    passed, summary = run_validation(
        artifacts_dir=artifacts_dir,
        expect_window_sensitivity=expect_window_sensitivity,
        output_json=summary_path,
    )
    if not passed:
        failed = [row for row in summary["checks"] if not row["passed"]]
        raise ValueError(f"Automated artifact validation failed: {failed}")
    print(f"Wrote output validation summary: {summary_path}")
    interpretation_path = write_interpretation(artifacts_dir=artifacts_dir, expect_window_sensitivity=expect_window_sensitivity)
    print(f"Wrote output interpretation: {interpretation_path}")


if __name__ == "__main__":
    main()
