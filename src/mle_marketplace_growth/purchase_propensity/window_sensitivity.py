"""Run 30/60/90-day label-window sensitivity for propensity modeling."""

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mle_marketplace_growth.purchase_propensity.helpers.data import _read_parquet_panel

ALLOWED_PREDICTION_WINDOWS = {30, 60, 90}
FIXED_WINDOWS = [30, 60, 90]
SENSITIVITY_MODELS = ["logistic_regression", "xgboost"]
DEFAULT_LOOKBACK_FOR_WINDOW_SWEEP = 90
LOOKBACK_SWEEP = [60, 90, 120]


def _run_train_eval(
    input_paths: list[Path],
    prediction_window_days: int,
    feature_lookback_days: int,
    output_dir: Path,
) -> dict:
    """Run train.py for both models on one (prediction_window, lookback) combo."""
    model_results = []
    for model_name in SENSITIVITY_MODELS:
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        command: list[str] = [
            sys.executable, "-m",
            "mle_marketplace_growth.purchase_propensity.train", *[arg for path in input_paths for arg in ("--input-path", str(path))],
            "--output-dir", str(model_output_dir),
            "--prediction-window-days", str(prediction_window_days),
            "--feature-lookback-days", str(feature_lookback_days),
            "--force-propensity-model", model_name,
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)

        metrics_path = model_output_dir / "train_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        validation_quality = metrics.get("validation_quality", {})
        model_results.append(
            {
                "model_name": str(metrics.get("selected_propensity_model_name", model_name)),
                "roc_auc": round(float(validation_quality.get("roc_auc", 0.0)), 6),
                "average_precision": round(float(validation_quality.get("average_precision", 0.0)), 6),
                "top_decile_lift": round(float(validation_quality.get("top_decile_lift", 0.0)), 6),
                "brier_score": round(float(validation_quality.get("brier_score", 1.0)), 6),
                "ece_10_bin": round(float(validation_quality.get("ece_10_bin", 1.0)), 6),
                "train_rows": int(metrics.get("train_rows", 0)),
                "validation_rows": int(metrics.get("validation_rows", 0)),
                "positive_rate_train": None,
                "positive_rate_validation": round(float(validation_quality.get("base_positive_rate", 0.0)), 6),
                "spend_cap_value": round(float(metrics.get("spend_cap_value", 0.0)), 6),
                "calibration_method": str(metrics.get("calibration_method", "sigmoid")),
            }
        )

    best_model = max(model_results, key=lambda row: row["average_precision"])["model_name"]
    return {"best_model": best_model, "model_results": model_results}


def _best_by_average_precision(model_results: list[dict]) -> dict:
    """Return the model row with highest average precision."""
    return max(model_results, key=lambda row: float(row.get("average_precision", 0.0)))


# ===== Label + Metric Utilities =====
def _load_positive_event_dates(path: Path) -> dict[str, list[date]]:
    events_df = _read_parquet_panel(path, allow_empty=True)
    if events_df.empty:
        return {}
    events_by_user: dict[str, list[date]] = {}
    for row in events_df.itertuples(index=False):
        user_id = str(row.user_id)
        if not user_id:
            continue
        if float(row.quantity) <= 0:
            continue
        event_day = date.fromisoformat(str(row.event_date))
        events_by_user.setdefault(user_id, []).append(event_day)
    for user_id in events_by_user:
        events_by_user[user_id].sort()
    return events_by_user


def _inter_purchase_gap_days(events_by_user: dict[str, list[date]]) -> list[int]:
    gaps = []
    for events in events_by_user.values():
        if len(events) < 2:
            continue
        for left, right in zip(events[:-1], events[1:], strict=True):
            gap_days = (right - left).days
            if gap_days > 0:
                gaps.append(gap_days)
    return gaps


# ===== Visualization =====
def _write_validation_dashboard(output: dict, output_path: Path) -> None:
    prediction_rows = output.get("window_sensitivity", [])
    lookback_rows = output.get("feature_window_validation", [])
    if not prediction_rows or not lookback_rows:
        return

    window_axis = [row["window_days"] for row in prediction_rows]
    best_window_metrics = [
        max(row["model_results"], key=lambda item: item["average_precision"]) for row in prediction_rows
    ]
    lookback_axis = [row["feature_lookback_days"] for row in lookback_rows]
    best_lookback_metrics = [
        max(row["model_results"], key=lambda item: item["average_precision"]) for row in lookback_rows
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metric_specs = [
        ("average_precision", "PR-AUC"),
        ("top_decile_lift", "Top-Decile Lift"),
        ("brier_score", "Brier Score"),
        ("ece_10_bin", "ECE (10-bin)"),
    ]
    for axis, (metric_key, title) in zip(axes.flat, metric_specs, strict=True):
        axis.plot(window_axis, [row[metric_key] for row in best_window_metrics], marker="o")
        axis.plot(lookback_axis, [row[metric_key] for row in best_lookback_metrics], marker="o")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("Window days")
    axes[0, 0].legend(["Prediction window", "Feature lookback"], loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote window validation dashboard: {output_path}")


def _resolve_inputs(args: argparse.Namespace) -> tuple[list[Path], Path]:
    """Resolve and validate feature/event inputs."""
    feature_paths = [Path(path) for path in args.input_path]
    events_path = Path(args.events_path)
    missing_features = [path for path in feature_paths if not path.exists()]
    if missing_features: raise FileNotFoundError(f"Input path not found: {missing_features[0]}")
    if not events_path.exists(): raise FileNotFoundError(f"Events path not found: {events_path}")
    return feature_paths, events_path


def _run_sensitivity_sweep(feature_paths: list[Path], sweep_root: Path) -> tuple[list[dict], list[dict]]:
    """Run prediction-window and lookback-window sensitivity sweeps."""
    eval_cache: dict[tuple[int, int], dict] = {}

    def _evaluate_combo(prediction_window_days: int, feature_lookback_days: int) -> dict:
        key = (prediction_window_days, feature_lookback_days)
        if key not in eval_cache:
            eval_cache[key] = _run_train_eval(
                input_paths=feature_paths,
                prediction_window_days=prediction_window_days,
                feature_lookback_days=feature_lookback_days,
                output_dir=sweep_root / f"prediction_{prediction_window_days}d_lookback_{feature_lookback_days}d",
            )
        return eval_cache[key]

    sensitivity_rows = []
    for window_days in FIXED_WINDOWS:
        combo_result = _evaluate_combo(window_days, DEFAULT_LOOKBACK_FOR_WINDOW_SWEEP)
        sensitivity_rows.append(
            {
                "window_days": window_days,
                "best_model_by_average_precision": combo_result["best_model"],
                "model_results": combo_result["model_results"],
            }
        )

    feature_window_validation = []
    prediction_window_for_feature_eval = 30
    for lookback_days in LOOKBACK_SWEEP:
        combo_result = _evaluate_combo(prediction_window_for_feature_eval, lookback_days)
        feature_window_validation.append(
            {
                "feature_lookback_days": lookback_days,
                "prediction_window_days_for_eval": prediction_window_for_feature_eval,
                "feature_profile_note": "direct feature schema",
                "best_model_by_average_precision": combo_result["best_model"],
                "model_results": combo_result["model_results"],
            }
        )
    return sensitivity_rows, feature_window_validation


def _build_output(
    feature_paths: list[Path],
    events_path: Path,
    inter_purchase_gap_days: list[int],
    sensitivity_rows: list[dict],
    feature_window_validation: list[dict],
) -> dict:
    """Build final JSON payload including freeze decision and diagnostics."""
    prediction_window_validation = []
    if inter_purchase_gap_days:
        for window_days in FIXED_WINDOWS:
            gap_array = np.asarray(inter_purchase_gap_days, dtype=float)
            coverage = float((gap_array <= window_days).mean())
            prediction_window_validation.append(
                {
                    "window_days": window_days,
                    "inter_purchase_gap_coverage": round(float(coverage), 6),
                }
            )

    best_prediction_window = max(sensitivity_rows, key=lambda row: _best_by_average_precision(row.get("model_results", [{"average_precision": 0.0}]))["average_precision"])
    best_lookback_window = max(feature_window_validation, key=lambda row: _best_by_average_precision(row.get("model_results", [{"average_precision": 0.0}]))["average_precision"])
    best_prediction_model = _best_by_average_precision(best_prediction_window["model_results"])
    best_lookback_model = _best_by_average_precision(best_lookback_window["model_results"])
    best_prediction_model_name, best_prediction_model_ap = str(best_prediction_model["model_name"]), float(best_prediction_model["average_precision"])
    best_lookback_model_name, best_lookback_model_ap = str(best_lookback_model["model_name"]), float(best_lookback_model["average_precision"])
    freeze_decision = {
        "selected_prediction_window_days": int(best_prediction_window["window_days"]),
        "selected_feature_lookback_days": int(best_lookback_window["feature_lookback_days"]),
        "selected_propensity_model_name": best_lookback_model_name,
        "selection_rule": "maximize_validation_average_precision",
        "selection_summary": {
            "prediction_window_model": best_prediction_model_name,
            "prediction_window_average_precision": round(best_prediction_model_ap, 6),
            "lookback_model": best_lookback_model_name,
            "lookback_average_precision": round(best_lookback_model_ap, 6),
        },
    }
    sorted_gaps = sorted(inter_purchase_gap_days)
    return {
        "input_paths": [str(path) for path in feature_paths],
        "events_path": str(events_path),
        "split_strategy": "out_of_time_10_1_1_train_dates=from_train_metrics",
        "spend_cap_quantile": 0.99,
        "calibration_method": "sigmoid",
        "note": "offline sensitivity only; not causal promotional incrementality evidence",
        "inter_purchase_distribution": {
            "gap_observation_count": len(inter_purchase_gap_days),
            "median_gap_days": sorted_gaps[len(sorted_gaps) // 2] if sorted_gaps else None,
            "mean_gap_days": round(float(np.mean(inter_purchase_gap_days)), 6)
            if inter_purchase_gap_days
            else None,
        },
        "prediction_window_validation": prediction_window_validation,
        "feature_window_validation": feature_window_validation,
        "window_sensitivity": sensitivity_rows,
        "freeze_decision": freeze_decision,
    }


# ===== Entry Point =====
def main() -> None:
    parser = argparse.ArgumentParser(description="Run 30/60/90-day sensitivity for propensity models.")
    parser.add_argument("--input-path", action="append", required=True, help="Path to strict training-panel parquet (repeat --input-path for multi-snapshot panel)")
    parser.add_argument("--events-path", default="data/silver/transactions_line_items/transactions_line_items.parquet", help="Path to silver transactions parquet (for recalculating labels by window)")
    parser.add_argument("--output-json", default="artifacts/purchase_propensity/window_sensitivity.json", help="Path to sensitivity summary JSON")
    parser.add_argument("--output-plot", default="artifacts/purchase_propensity/window_validation_dashboard.png", help="Path to window validation dashboard PNG")
    args = parser.parse_args()

    # This script orchestrates train.py runs; it does not train models directly.
    feature_paths, events_path = _resolve_inputs(args)

    windows = FIXED_WINDOWS
    if set(windows) != ALLOWED_PREDICTION_WINDOWS:
        raise ValueError("--windows must be exactly 30,60,90 for strict architecture alignment.")

    # ===== Diagnostics source =====
    events_by_user = _load_positive_event_dates(events_path)
    inter_purchase_gap_days = _inter_purchase_gap_days(events_by_user)

    # ===== Run Train-Eval Sweeps =====
    output_json_path = Path(args.output_json)
    sweep_root = output_json_path.parent / "_window_sensitivity_train_runs"
    sensitivity_rows, feature_window_validation = _run_sensitivity_sweep(feature_paths, sweep_root)
    output = _build_output(
        feature_paths=feature_paths,
        events_path=events_path,
        inter_purchase_gap_days=inter_purchase_gap_days,
        sensitivity_rows=sensitivity_rows,
        feature_window_validation=feature_window_validation,
    )
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote sensitivity summary: {output_json_path}")
    _write_validation_dashboard(output, Path(args.output_plot))


if __name__ == "__main__":
    main()
