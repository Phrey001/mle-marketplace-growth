"""Run the purchase propensity pipeline end-to-end from one command."""

# Special: orchestrates sensitivity -> train -> policy eval -> validation/report in one run.
# Suggested review order: run_pipeline.py -> train.py -> helpers/* -> policy_budget_evaluation.py -> validate_artifact_outputs.py -> window_sensitivity.py.

import argparse
from datetime import date
from pathlib import Path

from mle_marketplace_growth.helpers import cfg_required, generate_snapshot_dates, load_yaml_defaults
from mle_marketplace_growth.purchase_propensity.constants import (
    ALLOWED_FEATURE_LOOKBACK_WINDOWS,
    ALLOWED_PREDICTION_WINDOWS,
)
from mle_marketplace_growth.purchase_propensity.helpers.artifacts import (
    _cycle_artifacts_root,
    _offline_eval_paths,
    _report_paths,
)
from mle_marketplace_growth.purchase_propensity.policy_budget_evaluation import run_policy_budget_evaluation
from mle_marketplace_growth.purchase_propensity.train import run_training
from mle_marketplace_growth.purchase_propensity.validate_artifact_outputs import run_validation, write_interpretation
from mle_marketplace_growth.purchase_propensity.window_sensitivity import run_window_sensitivity


# ===== Entry Point =====
def main() -> None:
    """What: Orchestrate sensitivity, train, policy-eval, and artifact validation.
    Why: Provides one deterministic entrypoint for end-to-end offline evaluation.
    """
    # ===== CLI Args =====
    parser = argparse.ArgumentParser(description="Run purchase propensity pipeline end-to-end.")
    parser.add_argument("--config", required=True, help="YAML config file for pipeline arguments")
    args = parser.parse_args()

    # ===== Load Config =====
    cfg = load_yaml_defaults(args.config, "Engine config")
    # Config keys expected in cycle YAML.
    # Deterministic configs: always used (paths, dates, budgets).
    # Branching configs: control whether we run sensitivity or use fixed settings.
    # - window_selection_mode toggles sensitivity vs fixed path.
    # - force_propensity_model is required only in fixed mode; sensitivity mode derives it from window_sensitivity output.
    panel_end_date_raw = str(cfg_required(cfg, "panel_end_date"))
    prediction_window_days = int(cfg_required(cfg, "prediction_window_days"))  # allowed values: 30/60/90
    feature_lookback_days = int(cfg_required(cfg, "feature_lookback_days"))  # allowed values: 60/90/120
    window_selection_mode = str(cfg_required(cfg, "window_selection_mode"))  # allowed values: sensitivity|fixed
    force_propensity_model = cfg.get("force_propensity_model", None)
    budget = float(cfg_required(cfg, "budget"))
    cost_per_user = float(cfg_required(cfg, "cost_per_user"))
    output_root = Path("data")
    artifacts_dir = _cycle_artifacts_root(Path(args.config))
    offline_paths = _offline_eval_paths(artifacts_dir)
    report_paths = _report_paths(artifacts_dir)

    # ===== Validate Inputs =====
    panel_end_date = date.fromisoformat(panel_end_date_raw)
    train_as_of_dates = [snapshot.isoformat() for snapshot in generate_snapshot_dates(panel_end_date)]
    if window_selection_mode not in {"sensitivity", "fixed"}:
        raise ValueError("--window-selection-mode must be one of: sensitivity, fixed")
    if window_selection_mode == "fixed" and not force_propensity_model:  # fixed mode must supply a forced model
        raise ValueError("--force-propensity-model is required when window_selection_mode=fixed")
    if prediction_window_days not in ALLOWED_PREDICTION_WINDOWS: raise ValueError("--prediction-window-days must be one of: 30, 60, 90")
    if feature_lookback_days not in ALLOWED_FEATURE_LOOKBACK_WINDOWS: raise ValueError("--feature-lookback-days must be one of: 60, 90, 120")
    print(f"Window profile: prediction={prediction_window_days}d, feature_lookback={feature_lookback_days}d")

    # ===== Validate Inputs (Prebuilt Gold Required) =====
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
    # Mode-dependent behavior:
    # - sensitivity mode: run window_sensitivity.py and require its artifact in validation.
    # - fixed mode: skip sensitivity run and do not require sensitivity artifact.
    require_window_sensitivity_artifact = window_selection_mode == "sensitivity"
    if require_window_sensitivity_artifact:
        window_sensitivity_output = run_window_sensitivity(
            config_path=Path(args.config),
            feature_paths=train_paths,
            events_path=output_root / "silver" / "transactions_line_items" / "transactions_line_items.parquet",
            output_json_path=offline_paths.window_sensitivity_path,
            output_plot_path=offline_paths.window_validation_plot_path,
        )
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
    run_training(
        config_path=Path(args.config),
        output_dir=offline_paths.root,
        prediction_window_days=frozen_prediction_window_days,
        feature_lookback_days=frozen_feature_lookback_days,
        force_propensity_model=frozen_propensity_model,
    )
    # ===== Offline Policy Evaluation (Validation + Test) =====
    run_policy_budget_evaluation(
        scores_csv=offline_paths.validation_predictions_path,
        output_json=offline_paths.validation_policy_path,
        budget=budget,
        cost_per_user=cost_per_user,
        prediction_window_days=frozen_prediction_window_days,
    )
    run_policy_budget_evaluation(
        scores_csv=offline_paths.test_predictions_path,
        output_json=offline_paths.test_policy_path,
        budget=budget,
        cost_per_user=cost_per_user,
        prediction_window_days=frozen_prediction_window_days,
    )
    # ===== Validate + Write Outputs =====
    passed, summary = run_validation(
        artifacts_dir=offline_paths.root,
        expect_window_sensitivity=require_window_sensitivity_artifact,
        output_json=report_paths.validation_summary_path,
    )
    if not passed:
        failed = [row for row in summary["checks"] if not row["passed"]]
        raise ValueError(f"Automated artifact validation failed: {failed}")
    print(f"Wrote output validation summary: {report_paths.validation_summary_path}")
    interpretation_path = write_interpretation(
        artifacts_dir=offline_paths.root,
        output_md=report_paths.interpretation_path,
        expect_window_sensitivity=require_window_sensitivity_artifact,
    )
    print(f"Wrote output interpretation: {interpretation_path}")

if __name__ == "__main__":
    main()
