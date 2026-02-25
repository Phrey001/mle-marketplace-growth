"""Run the purchase propensity pipeline end-to-end from one command."""

import argparse
import calendar
import csv
import subprocess
import sys
from datetime import date
from pathlib import Path

from mle_marketplace_growth.purchase_propensity.validate_outputs import run_validation, write_interpretation

ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}
ALLOWED_FEATURE_LOOKBACK_WINDOWS = {60, 90, 120}


# ===== Config + Shell Helpers =====
def _read_config_file(path: Path) -> dict:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Config file must use .yaml or .yml")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required for YAML configs. Install with: pip install pyyaml") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a key-value object")
    return payload


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


# ===== Path + Date Helpers =====
def _train_dataset_path(output_root: Path, as_of_date: str) -> Path:
    return (
        output_root
        / "gold"
        / "feature_store"
        / "purchase_propensity"
        / "propensity_train_dataset"
        / f"as_of_date={as_of_date}"
        / "propensity_train_dataset.csv"
    )


def _parse_date(value: str, arg_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be YYYY-MM-DD, got: {value}") from exc


def _add_month(current: date) -> date:
    year = current.year + (1 if current.month == 12 else 0)
    month = 1 if current.month == 12 else current.month + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _generate_snapshot_dates(start_date: date, end_date: date) -> list[str]:
    if end_date < start_date:
        raise ValueError("--train-end-date must be on or after --train-start-date")

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.isoformat())
        current = _add_month(current)
    return dates


# ===== Pipeline Steps =====
def _build_snapshot(input_csv: Path, output_root: Path, as_of_date: str) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "mle_marketplace_growth.feature_store.build",
            "--build-engines",
            "purchase_propensity",
            "--input-csv",
            str(input_csv),
            "--output-root",
            str(output_root),
            "--as-of-date",
            as_of_date,
        ]
    )


def _merge_train_datasets(train_paths: list[Path], merged_output_path: Path) -> None:
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with merged_output_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = None
        for path in train_paths:
            if not path.exists():
                raise FileNotFoundError(f"Training dataset not found: {path}")
            with path.open("r", encoding="utf-8", newline="") as in_file:
                reader = csv.DictReader(in_file)
                if not reader.fieldnames:
                    raise ValueError(f"Missing header in training dataset: {path}")
                if writer is None:
                    writer = csv.DictWriter(out_file, fieldnames=reader.fieldnames)
                if not wrote_header:
                    writer.writeheader()
                    wrote_header = True
                for row in reader:
                    writer.writerow(row)


# ===== Entry Point =====
def main() -> None:
    # ===== Parse Config Defaults =====
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional YAML config file for pipeline arguments")
    pre_args, remaining_argv = pre_parser.parse_known_args()
    if pre_args.config:
        config_path = Path(pre_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config_defaults = _read_config_file(config_path)
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
    parser.add_argument(
        "--train-frequency",
        choices=["monthly"],
        default=cfg("train_frequency", "monthly"),
        help="Frequency for generated training snapshots (monthly only)",
    )
    parser.add_argument("--score-as-of-date", default=cfg("score_as_of_date", "2011-11-09"), help="Feature snapshot date used for scoring/evaluation")
    parser.add_argument(
        "--validation-mode",
        choices=["hash", "out_of_time"],
        default=cfg("validation_mode", "hash"),
        help="Validation strategy for training",
    )
    parser.add_argument("--validation-rate", type=float, default=float(cfg("validation_rate", 0.2)), help="Holdout fraction used for validation split")
    parser.add_argument(
        "--prediction-window-days",
        type=int,
        default=int(cfg("prediction_window_days", 30)),
        help="Allowed values: 30, 60, 90 (current feature-store SQL is implemented for 30-day labels).",
    )
    parser.add_argument(
        "--feature-lookback-days",
        type=int,
        default=int(cfg("feature_lookback_days", 90)),
        help="Allowed values: 60, 90, 120 (current feature-store SQL is implemented for 90-day features).",
    )
    parser.add_argument("--target-rate", type=float, default=cfg("target_rate", 0.2), help="Targeting share for policy comparison")
    parser.add_argument("--budget", type=float, default=cfg("budget", 5000.0), help="Budget for offline policy evaluation")
    parser.add_argument("--cost-per-user", type=float, default=cfg("cost_per_user", 5.0), help="Cost per targeted user for offline policy evaluation")
    parser.add_argument("--sensitivity-as-of-date", default=cfg("sensitivity_as_of_date", "2011-09-09"), help="Feature snapshot date used for window sensitivity")
    args = parser.parse_args(remaining_argv)

    # ===== Validate Inputs =====
    input_csv = Path(args.input_csv)
    output_root = Path(args.output_root)
    artifacts_dir = Path(args.artifacts_dir)
    use_generated_dates = bool(args.train_start_date or args.train_end_date)
    if not use_generated_dates and not args.train_as_of_dates.strip():
        args.train_as_of_dates = "2011-11-09"
    if use_generated_dates and args.train_as_of_dates.strip():
        raise ValueError("Use either --train-as-of-dates or --train-start-date/--train-end-date, not both")
    if use_generated_dates:
        if not args.train_start_date or not args.train_end_date:
            raise ValueError("Both --train-start-date and --train-end-date are required together")
        start_date = _parse_date(args.train_start_date, "--train-start-date")
        end_date = _parse_date(args.train_end_date, "--train-end-date")
        train_as_of_dates = _generate_snapshot_dates(start_date, end_date)
    else:
        train_as_of_dates = [value.strip() for value in args.train_as_of_dates.split(",") if value.strip()]

    if not train_as_of_dates:
        raise ValueError("--train-as-of-dates must include at least one date")
    if args.validation_mode == "out_of_time" and len(train_as_of_dates) < 2:
        raise ValueError("--validation-mode out_of_time requires at least 2 --train-as-of-dates")
    if not 0.0 < args.validation_rate < 1.0:
        raise ValueError("--validation-rate must be between 0 and 1")
    if args.prediction_window_days not in ALLOWED_PREDICTION_WINDOWS:
        raise ValueError("--prediction-window-days must be one of: 30, 60, 90")
    if args.feature_lookback_days not in ALLOWED_FEATURE_LOOKBACK_WINDOWS:
        raise ValueError("--feature-lookback-days must be one of: 60, 90, 120")
    if args.prediction_window_days != 30:
        raise ValueError(
            "Current feature-store SQL/materialization is implemented for 30-day labels only. "
            "60/90-day windows are reserved but not wired into the main pipeline yet."
        )
    if args.feature_lookback_days != 90:
        raise ValueError(
            "Current feature-store SQL/materialization is implemented for 90-day feature lookback only. "
            "Other lookback windows are reserved but not wired into the main pipeline yet."
        )
    print(f"Window profile: prediction={args.prediction_window_days}d, feature_lookback={args.feature_lookback_days}d")

    # ===== Build Feature-Store Snapshots =====
    snapshots_to_build = set(train_as_of_dates)
    snapshots_to_build.add(args.score_as_of_date)
    snapshots_to_build.add(args.sensitivity_as_of_date)
    for as_of_date in sorted(snapshots_to_build):
        _build_snapshot(input_csv, output_root, as_of_date)

    # ===== Build Training Input =====
    train_paths = [_train_dataset_path(output_root, as_of_date) for as_of_date in train_as_of_dates]
    if len(train_paths) == 1:
        train_input_csv = train_paths[0]
    else:
        train_input_csv = artifacts_dir / "_tmp" / "propensity_train_dataset_merged.csv"
        _merge_train_datasets(train_paths, train_input_csv)

    # ===== Train + Predict + Evaluate =====
    _run(
        [
            sys.executable,
            "-m",
            "mle_marketplace_growth.purchase_propensity.train",
            "--input-csv",
            str(train_input_csv),
            "--output-dir",
            str(artifacts_dir),
            "--validation-mode",
            args.validation_mode,
            "--validation-rate",
            str(args.validation_rate),
            "--target-rate",
            str(args.target_rate),
        ]
    )

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
    evaluation_json_path = artifacts_dir / "evaluation.json"
    evaluation_plot_path = artifacts_dir / "evaluation_policy_comparison.png"
    evaluation_diag_path = artifacts_dir / "evaluation_model_diagnostics.png"
    offline_eval_json_path = artifacts_dir / "offline_policy_evaluation.json"
    offline_eval_plot_path = artifacts_dir / "offline_policy_evaluation_budget_curve.png"

    _run(
        [
            sys.executable,
            "-m",
            "mle_marketplace_growth.purchase_propensity.predict",
            "--input-csv",
            str(user_features_path),
            "--model-path",
            str(model_path),
            "--output-csv",
            str(prediction_scores_path),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "mle_marketplace_growth.purchase_propensity.evaluate",
            "--scores-csv",
            str(validation_predictions_path),
            "--output-json",
            str(evaluation_json_path),
            "--output-plot",
            str(evaluation_plot_path),
            "--output-diagnostics-plot",
            str(evaluation_diag_path),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "mle_marketplace_growth.purchase_propensity.offline_policy_evaluation",
            "--scores-csv",
            str(prediction_scores_path),
            "--output-json",
            str(offline_eval_json_path),
            "--output-plot",
            str(offline_eval_plot_path),
            "--budget",
            str(args.budget),
            "--cost-per-user",
            str(args.cost_per_user),
        ]
    )

    # ===== Sensitivity Run =====
    _run(
        [
            sys.executable,
            "-m",
            "mle_marketplace_growth.purchase_propensity.window_sensitivity",
            "--input-csv",
            str(_train_dataset_path(output_root, args.sensitivity_as_of_date)),
            "--events-csv",
            str(output_root / "silver" / "transactions_line_items" / "transactions_line_items.csv"),
            "--output-json",
            str(artifacts_dir / "window_sensitivity.json"),
            "--output-plot",
            str(artifacts_dir / "window_validation_dashboard.png"),
        ]
    )

    # ===== Validation + Interpretation =====
    summary_path = artifacts_dir / "output_validation_summary.json"
    passed, summary = run_validation(
        artifacts_dir=artifacts_dir,
        expect_window_sensitivity=True,
        output_json=summary_path,
    )
    if not passed:
        failed = [row for row in summary["checks"] if not row["passed"]]
        raise ValueError(f"Automated artifact validation failed: {failed}")
    print(f"Wrote output validation summary: {summary_path}")
    interpretation_path = write_interpretation(artifacts_dir=artifacts_dir)
    print(f"Wrote output interpretation: {interpretation_path}")


if __name__ == "__main__":
    main()
