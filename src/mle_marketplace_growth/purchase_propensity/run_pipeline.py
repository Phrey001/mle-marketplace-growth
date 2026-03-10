"""Run the purchase propensity pipeline end-to-end from one command."""

# Special: orchestrates sensitivity -> train -> policy eval -> validation/report in one run.
# Suggested review order: run_pipeline.py -> train.py -> helpers/* -> policy_budget_evaluation.py -> validate_outputs.py -> window_sensitivity.py.

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from dateutil.relativedelta import relativedelta
from mle_marketplace_growth.feature_store.build_helpers import load_yaml_defaults
from mle_marketplace_growth.purchase_propensity.validate_outputs import run_validation, write_interpretation

ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}
ALLOWED_FEATURE_LOOKBACK_WINDOWS = {60, 90, 120}


# ===== Path + Date Helpers =====
def _generate_snapshot_dates(panel_end_date: date) -> list[str]:
    # 12 inclusive snapshots: offsets -11..0 from the end date.
    snapshots = [panel_end_date + relativedelta(months=offset) for offset in range(-11, 1)]
    if snapshots[-1] != panel_end_date.isoformat():
        raise ValueError("Derived monthly snapshot panel does not end on --panel-end-date")
    return [snapshot.isoformat() for snapshot in snapshots]


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

    # Config keys expected in cycle YAML (with defaults where optional).
    # Deterministic configs: always used (paths, dates, budgets).
    # Branching configs: control whether we run sensitivity or use fixed settings.
    # - window_selection_mode toggles sensitivity vs fixed path.
    # - force_propensity_model is required only in fixed mode.
    panel_end_date_raw = cfg("panel_end_date", None)
    output_root = Path(cfg("output_root", "data"))  # default path
    prediction_window_days = int(cfg("prediction_window_days", 30))  # allowed values: 30/60/90
    feature_lookback_days = int(cfg("feature_lookback_days", 90))  # allowed values: 60/90/120
    window_selection_mode = cfg("window_selection_mode", "sensitivity")  # allowed values: sensitivity|fixed
    force_propensity_model = cfg("force_propensity_model", None)
    budget = float(cfg("budget", 5000.0))
    cost_per_user = float(cfg("cost_per_user", 5.0))

    # Optional key not present in cycle YAMLs (falls back to default).
    # Default artifacts dir uses the config filename stem (stem removes .extension; e.g. artifacts/purchase_propensity/cycle_initial/).
    artifacts_dir = Path(cfg("artifacts_dir", str(Path("artifacts/purchase_propensity") / Path(args.config).stem)))

    # ===== Artifacts subfolders ===== 
    offline_eval_dir = artifacts_dir / "offline_eval"  
    report_dir = artifacts_dir / "report"

    # ===== Validate Inputs =====
    if not panel_end_date_raw: raise ValueError("--panel-end-date is required")
    panel_end_date = date.fromisoformat(panel_end_date_raw)
    train_as_of_dates = _generate_snapshot_dates(panel_end_date)
    if window_selection_mode == "fixed" and not force_propensity_model:  # fixed mode must supply a forced model
        raise ValueError("--force-propensity-model is required when window_selection_mode=fixed")
    if prediction_window_days not in ALLOWED_PREDICTION_WINDOWS: raise ValueError("--prediction-window-days must be one of: 30, 60, 90")
    if feature_lookback_days not in ALLOWED_FEATURE_LOOKBACK_WINDOWS: raise ValueError("--feature-lookback-days must be one of: 60, 90, 120")
    print(f"Window profile: prediction={prediction_window_days}d, feature_lookback={feature_lookback_days}d")

    # ===== Build Training Input (Prebuilt Gold Required) =====
    train_paths = [
        output_root / "gold" / "feature_store" / "purchase_propensity" / "propensity_train_dataset"
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
    # ===== Structural Decision: Sensitivity Freeze or Fixed Config =====
    expect_window_sensitivity = window_selection_mode == "sensitivity"
    if expect_window_sensitivity:
        _run_module(
            "mle_marketplace_growth.purchase_propensity.window_sensitivity",
            "--panel-root", output_root / "gold" / "feature_store" / "purchase_propensity" / "propensity_train_dataset",
            "--panel-end-date", panel_end_date.isoformat(),
            "--events-path", output_root / "silver" / "transactions_line_items" / "transactions_line_items.parquet",
            "--output-json", offline_eval_dir / "window_sensitivity.json",
            "--output-plot", offline_eval_dir / "window_validation_dashboard.png",
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
    # Build CLI args separately so we can append the optional forced model cleanly.
    # Repeat --input-path for multi-snapshot panel.
    train_args: list[str | Path | int] = [
        *[arg for path in train_paths for arg in ("--input-path", path)],
        "--output-dir", offline_eval_dir,
        "--prediction-window-days", frozen_prediction_window_days,
        "--feature-lookback-days", frozen_feature_lookback_days,
    ]
    if frozen_propensity_model: train_args.extend(["--force-propensity-model", frozen_propensity_model])
    _run_module("mle_marketplace_growth.purchase_propensity.train", *train_args)
    # ===== Offline Policy Evaluation (Validation + Test) =====
    # Build policy_eval_args once since it is reused for validation and test runs.
    policy_eval_args = [
        "--budget", budget,
        "--cost-per-user", cost_per_user,
        "--prediction-window-days", frozen_prediction_window_days,
    ]
    _run_module(
        "mle_marketplace_growth.purchase_propensity.policy_budget_evaluation",
        "--scores-csv", offline_eval_dir / "validation_predictions.csv",
        "--output-json", offline_eval_dir / "offline_policy_budget_validation.json",
        *policy_eval_args,
    )
    _run_module(
        "mle_marketplace_growth.purchase_propensity.policy_budget_evaluation",
        "--scores-csv", offline_eval_dir / "test_predictions.csv",
        "--output-json", offline_eval_dir / "offline_policy_budget_test.json",
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
