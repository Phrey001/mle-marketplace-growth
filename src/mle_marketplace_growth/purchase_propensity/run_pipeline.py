"""Run the purchase propensity pipeline end-to-end from one command."""

import argparse
import calendar
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


def _run_module(module: str, *args: object) -> None:
    command = [sys.executable, "-m", module, *map(str, args)]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


# ===== Entry Point =====
def main() -> None:
    parser = argparse.ArgumentParser(description="Run purchase propensity pipeline end-to-end.")
    parser.add_argument("--config", required=True, help="YAML config file for pipeline arguments")
    args = parser.parse_args()
    cfg = load_yaml_defaults(args.config, "Engine config").get
    output_root = Path(cfg("output_root", "data"))
    artifacts_dir = Path(cfg("artifacts_dir", str(Path("artifacts/purchase_propensity") / Path(args.config).stem)))
    panel_end_date_raw = cfg("panel_end_date", None)
    prediction_window_days = int(cfg("prediction_window_days", 30))
    feature_lookback_days = int(cfg("feature_lookback_days", 90))
    window_selection_mode = cfg("window_selection_mode", "sensitivity")
    force_propensity_model = cfg("force_propensity_model", None)
    budget = float(cfg("budget", 5000.0))
    cost_per_user = float(cfg("cost_per_user", 5.0))

    # ===== Validate Inputs =====
    offline_eval_dir = artifacts_dir / "offline_eval"
    report_dir = artifacts_dir / "report"
    if not panel_end_date_raw:
        raise ValueError("--panel-end-date is required")
    panel_end_date = date.fromisoformat(panel_end_date_raw)
    train_as_of_dates = _generate_snapshot_dates(panel_end_date)
    if prediction_window_days not in ALLOWED_PREDICTION_WINDOWS: raise ValueError("--prediction-window-days must be one of: 30, 60, 90")
    if feature_lookback_days not in ALLOWED_FEATURE_LOOKBACK_WINDOWS: raise ValueError("--feature-lookback-days must be one of: 60, 90, 120")
    print(f"Window profile: prediction={prediction_window_days}d, feature_lookback={feature_lookback_days}d")

    # ===== Build Training Input (Prebuilt Gold Required) =====
    train_paths = [
        output_root
        / "gold"
        / "feature_store"
        / "purchase_propensity"
        / "propensity_train_dataset"
        / f"as_of_date={as_of_date}"
        / "propensity_train_dataset.parquet"
        for as_of_date in train_as_of_dates
    ]
    missing_train_paths = [path for path in train_paths if not path.exists()]
    if missing_train_paths:
        raise FileNotFoundError(
            "Missing prebuilt purchase-propensity gold datasets. "
            "Build them first with `mle_marketplace_growth.feature_store.build_gold_purchase_propensity` "
            f"(example missing path: {missing_train_paths[0]})."
        )
    train_input_paths = train_paths

    # ===== Structural Decision: Sensitivity Freeze or Fixed Config =====
    expect_window_sensitivity = window_selection_mode == "sensitivity"
    if expect_window_sensitivity:
        _run_module(
            "mle_marketplace_growth.purchase_propensity.window_sensitivity",
            "--panel-root",
            output_root / "gold" / "feature_store" / "purchase_propensity" / "propensity_train_dataset",
            "--panel-end-date",
            panel_end_date.isoformat(),
            "--events-path",
            output_root / "silver" / "transactions_line_items" / "transactions_line_items.parquet",
            "--output-json",
            offline_eval_dir / "window_sensitivity.json",
            "--output-plot",
            offline_eval_dir / "window_validation_dashboard.png",
        )
        window_sensitivity_output = json.loads((offline_eval_dir / "window_sensitivity.json").read_text(encoding="utf-8"))
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
        frozen_prediction_window_days = int(prediction_window_days)
        frozen_feature_lookback_days = int(feature_lookback_days)
        frozen_propensity_model = force_propensity_model
        print(
            "Frozen decision from fixed config:",
            f"prediction_window={frozen_prediction_window_days}d,",
            f"feature_lookback={frozen_feature_lookback_days}d,",
            f"propensity_model={frozen_propensity_model}",
        )

    # ===== Train + Offline Evaluate =====
    train_args: list[str | Path | int] = [
        *[arg for path in train_input_paths for arg in ("--input-path", path)],
        "--output-dir",
        offline_eval_dir,
        "--prediction-window-days",
        frozen_prediction_window_days,
        "--feature-lookback-days",
        frozen_feature_lookback_days,
    ]
    if frozen_propensity_model: train_args.extend(["--force-propensity-model", frozen_propensity_model])
    _run_module("mle_marketplace_growth.purchase_propensity.train", *train_args)
    validation_predictions_path = offline_eval_dir / "validation_predictions.csv"
    test_predictions_path = offline_eval_dir / "test_predictions.csv"
    budget_eval_validation_json_path = offline_eval_dir / "offline_policy_budget_validation.json"
    budget_eval_test_json_path = offline_eval_dir / "offline_policy_budget_test.json"

    # ===== Offline Policy Evaluation (Validation + Test) =====
    policy_eval_args = [
        "--budget",
        budget,
        "--cost-per-user",
        cost_per_user,
        "--prediction-window-days",
        frozen_prediction_window_days,
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
    summary_path = report_dir / "output_validation_summary.json"
    passed, summary = run_validation(
        artifacts_dir=offline_eval_dir,
        expect_window_sensitivity=expect_window_sensitivity,
        output_json=summary_path,
    )
    if not passed:
        failed = [row for row in summary["checks"] if not row["passed"]]
        raise ValueError(f"Automated artifact validation failed: {failed}")
    print(f"Wrote output validation summary: {summary_path}")
    interpretation_path = write_interpretation(
        artifacts_dir=offline_eval_dir,
        output_md=report_dir / "output_interpretation.md",
        expect_window_sensitivity=expect_window_sensitivity,
    )
    print(f"Wrote output interpretation: {interpretation_path}")

if __name__ == "__main__":
    main()
