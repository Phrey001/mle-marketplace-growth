"""Run the purchase propensity pipeline end-to-end from one command."""

import argparse
import calendar
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from mle_marketplace_growth.feature_store.build_helpers import load_yaml_defaults
from mle_marketplace_growth.purchase_propensity.validate_outputs import run_validation, write_interpretation

ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}
ALLOWED_FEATURE_LOOKBACK_WINDOWS = {60, 90, 120}


# ===== Path + Date Helpers =====
def _add_month(current: date) -> date:
    year = current.year + (1 if current.month == 12 else 0)
    month = 1 if current.month == 12 else current.month + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _shift_month(current: date, delta_months: int) -> date:
    month_index = current.month - 1 + delta_months
    year = current.year + month_index // 12
    month = month_index % 12 + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _generate_snapshot_dates(panel_end_date: date) -> list[str]:
    start_date = _shift_month(panel_end_date, -11)
    snapshots = []
    current = start_date
    for _ in range(12):
        snapshots.append(current.isoformat())
        current = _add_month(current)
    if snapshots[-1] != panel_end_date.isoformat():
        raise ValueError("Derived monthly snapshot panel does not end on --panel-end-date")
    return snapshots


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


def _default_artifacts_dir_from_config(config_path_value: str | None, base_dir: str) -> str:
    if not config_path_value:
        return base_dir
    return str(Path(base_dir) / Path(config_path_value).stem)


def _resolve_from_cfg(args: argparse.Namespace, cfg_get, name: str, default, cast=None):
    value = getattr(args, name)
    value = cfg_get(name, default) if value is None else value
    return cast(value) if cast else value

# ===== Entry Point =====
def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Run purchase propensity pipeline end-to-end.")
    parser.add_argument("--config", required=True, help="YAML config file for pipeline arguments")

    parser.add_argument("--output-root", default=None, help="Root containing prebuilt purchase-propensity gold feature-store datasets")
    parser.add_argument("--artifacts-dir", default=None, help="Output directory for model/evaluation artifacts")
    parser.add_argument("--panel-end-date", default=None, help="End anchor (YYYY-MM-DD) for strict 12-month panel; derived split is 10 train / 1 val / 1 test.")
    parser.add_argument("--prediction-window-days", type=int, default=None, help="Allowed values: 30, 60, 90 (default baseline: 30).")
    parser.add_argument("--feature-lookback-days", type=int, default=None, help="Allowed values: 60, 90, 120 (default baseline: 90).")
    parser.add_argument("--window-selection-mode", choices=["sensitivity", "fixed"], default=None, help="`sensitivity`: select structural decisions from window_sensitivity output; `fixed`: use config values directly.")
    parser.add_argument("--force-propensity-model", choices=["logistic_regression", "xgboost"], default=None, help="Only used when --window-selection-mode=fixed.")
    parser.add_argument("--budget", type=float, default=None, help="Budget for offline policy evaluation")
    parser.add_argument("--cost-per-user", type=float, default=None, help="Cost per targeted user for offline policy evaluation")
    args = parser.parse_args()
    cfg = load_yaml_defaults(args.config, "Engine config").get
    for name, default, cast in [
        ("output_root", "data", None),
        ("artifacts_dir", _default_artifacts_dir_from_config(args.config, "artifacts/purchase_propensity"), None),
        ("panel_end_date", None, None),
        ("prediction_window_days", 30, int),
        ("feature_lookback_days", 90, int),
        ("window_selection_mode", "sensitivity", None),
        ("force_propensity_model", None, None),
        ("budget", 5000.0, float),
        ("cost_per_user", 5.0, float),
    ]:
        setattr(args, name, _resolve_from_cfg(args, cfg, name, default, cast))

    # ===== Validate Inputs =====
    output_root = Path(args.output_root)
    artifacts_dir = Path(args.artifacts_dir)
    if not args.panel_end_date:
        raise ValueError("--panel-end-date is required")
    panel_end_date = date.fromisoformat(args.panel_end_date)
    train_as_of_dates = _generate_snapshot_dates(panel_end_date)
    score_as_of_date = train_as_of_dates[-1]
    if args.prediction_window_days not in ALLOWED_PREDICTION_WINDOWS: raise ValueError("--prediction-window-days must be one of: 30, 60, 90")
    if args.feature_lookback_days not in ALLOWED_FEATURE_LOOKBACK_WINDOWS: raise ValueError("--feature-lookback-days must be one of: 60, 90, 120")
    print(f"Window profile: prediction={args.prediction_window_days}d, feature_lookback={args.feature_lookback_days}d")

    # ===== Build Training Input (Prebuilt Gold Required) =====
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
    missing_train_paths = [path for path in train_paths if not path.exists()]
    if missing_train_paths:
        raise FileNotFoundError(
            "Missing prebuilt purchase-propensity gold datasets. "
            "Build them first with `mle_marketplace_growth.feature_store.build_gold_purchase_propensity` "
            f"(example missing path: {missing_train_paths[0]})."
        )
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
        / f"as_of_date={score_as_of_date}"
        / "user_features_asof.csv"
    )
    if not user_features_path.exists():
        raise FileNotFoundError(
            "Missing prebuilt user features for scoring. "
            "Build gold first with `mle_marketplace_growth.feature_store.build_gold_purchase_propensity` "
            f"(expected path: {user_features_path})."
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

    # ===== Offline Policy Evaluation (Validation + Test) =====
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
