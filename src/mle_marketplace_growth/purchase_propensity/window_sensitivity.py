"""Run 30/60/90-day label-window sensitivity for propensity modeling."""

import argparse
import traceback
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mle_marketplace_growth.helpers import cfg_required, generate_snapshot_dates, load_yaml_defaults, read_json
from mle_marketplace_growth.purchase_propensity.constants import (
    ALLOWED_FEATURE_LOOKBACK_WINDOWS,
    ALLOWED_PREDICTION_WINDOWS,
    DEFAULT_LOOKBACK_FOR_WINDOW_SWEEP,
    SENSITIVITY_MODEL_NAMES,
    SPEND_CAP_QUANTILE,
)
from mle_marketplace_growth.purchase_propensity.helpers.artifacts import (
    _cycle_artifacts_root,
    _write_window_sensitivity_artifact,
    _write_window_validation_dashboard,
)
from mle_marketplace_growth.purchase_propensity.helpers.data import _read_parquet_panel
from mle_marketplace_growth.purchase_propensity.train import run_training

FIXED_WINDOWS = sorted(ALLOWED_PREDICTION_WINDOWS)
SENSITIVITY_MODELS = list(SENSITIVITY_MODEL_NAMES)
LOOKBACK_SWEEP = sorted(ALLOWED_FEATURE_LOOKBACK_WINDOWS)


def _load_train_metrics(metrics_path: Path) -> dict:
    """What: Load one `train_metrics.json` artifact from a train.py run.
    Why: Reuses train.py outputs as the source of truth for sweep comparison.
    """
    return read_json(metrics_path)


def _build_model_result_row(metrics: dict, model_name: str) -> dict:
    """What: Build one normalized model-result row from train.py metrics.
    Why: Keeps sweep aggregation schema consistent across model runs.
    """
    validation_quality = metrics.get("validation_quality", {})
    return {
        "status": "ok",
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


def _failed_model_result_row(model_name: str, error_message: str) -> dict:
    """What: Build a placeholder row when one model/window combo is untrainable.
    Why: Lets sensitivity continue and produce a complete artifact instead of aborting.
    """
    return {
        "status": "invalid",
        "model_name": model_name,
        "roc_auc": 0.0,
        "average_precision": 0.0,
        "top_decile_lift": 0.0,
        "brier_score": 1.0,
        "ece_10_bin": 1.0,
        "train_rows": 0,
        "validation_rows": 0,
        "positive_rate_train": None,
        "positive_rate_validation": 0.0,
        "spend_cap_value": 0.0,
        "calibration_method": "sigmoid",
        "train_error": error_message,
    }


def _compact_error(error: Exception) -> str:
    """What: Build a short, artifact-friendly error string from a failed in-process run.
    Why: Keeps sensitivity JSON readable while preserving root failure context.
    """
    message = str(error).strip()
    if message:
        return f"{type(error).__name__}: {message}"
    tail = traceback.format_exc().strip().splitlines()
    return tail[-1] if tail else f"{type(error).__name__}: unknown error"


def _run_train_eval(
    config_path: Path,
    input_paths: list[Path],
    prediction_window_days: int,
    feature_lookback_days: int,
    output_dir: Path,
) -> dict:
    """What: Run train.py for both propensity models on one window/lookback combo.
    Why: Compares model quality under the same data slice and feature profile.
    """
    model_results = []
    for model_name in SENSITIVITY_MODELS:
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_training(
                config_path=config_path,
                input_paths=input_paths,
                output_dir=model_output_dir,
                prediction_window_days=prediction_window_days,
                feature_lookback_days=feature_lookback_days,
                force_propensity_model=model_name,
            )
        except Exception as error:
            model_results.append(_failed_model_result_row(model_name, _compact_error(error)))
            continue

        metrics_path = model_output_dir / "train_metrics.json"
        metrics = _load_train_metrics(metrics_path)
        model_results.append(_build_model_result_row(metrics, model_name))

    best_model = max(model_results, key=lambda row: row["average_precision"])["model_name"]
    return {"best_model": best_model, "model_results": model_results}


def _best_by_average_precision(model_results: list[dict]) -> dict:
    """What: Return the model result row with highest validation average precision.
    Why: Uses one deterministic selection rule across all sweep outputs.
    """
    return max(model_results, key=lambda row: float(row.get("average_precision", 0.0)))


# ===== Label + Metric Utilities =====
def _inter_purchase_gap_days(events_path: Path) -> list[int]:
    """What: Compute day differences between consecutive positive-purchase events per user.
    Why: Summarizes repurchase cadence as a diagnostic for prediction-window selection.
    Downstream use in this repo:
    - `_build_output` converts these gaps into `prediction_window_validation`
      `inter_purchase_gap_coverage` stats.
    - Coverage for a candidate window is computed as
      mean(inter_purchase_gap_days <= window_days).
    - Those stats are written to `window_sensitivity.json` as supporting evidence for
      the freeze decision (alongside validation model metrics).

    Example:
    If one user has purchases on 2026-01-01, 2026-01-10, and 2026-01-25,
    this contributes gaps [9, 15] days.
    """
    events_df = _read_parquet_panel(events_path, allow_empty=True)
    if events_df.empty:
        return []

    # Keep only positive-quantity events and the fields needed for gap computation.
    filtered = events_df.loc[events_df["quantity"].astype(float) > 0, ["user_id", "event_date"]].copy()
    # Normalize user ids and drop blank identifiers to avoid invalid groups.
    filtered["user_id"] = filtered["user_id"].astype(str)
    filtered = filtered.loc[filtered["user_id"] != ""]
    # Parse event dates and normalize to day-level timestamps.
    filtered["event_day"] = pd.to_datetime(filtered["event_date"], errors="coerce").dt.normalize()
    filtered = filtered.dropna(subset=["event_day"])

    # Sort by user/day so per-user previous-event shift yields consecutive gaps.
    filtered = filtered.sort_values(["user_id", "event_day"], kind="stable")
    previous_event_day = filtered.groupby("user_id")["event_day"].shift(1)
    # Convert date deltas to integer days and keep strictly positive gaps only.
    gap_days = (filtered["event_day"] - previous_event_day).dt.days
    return gap_days.loc[gap_days > 0].astype(int).tolist()


def _run_sensitivity_sweep(config_path: Path, feature_paths: list[Path], sweep_root: Path) -> tuple[list[dict], list[dict]]:
    """What: Run prediction-window and feature-lookback sensitivity sweeps.
    Why: Produces comparable validation outputs for freeze-decision selection.
    """
    eval_cache: dict[tuple[int, int], dict] = {}

    def _evaluate_combo(prediction_window_days: int, feature_lookback_days: int) -> dict:
        key = (prediction_window_days, feature_lookback_days)
        if key not in eval_cache:
            eval_cache[key] = _run_train_eval(
                config_path=config_path,
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
    """What: Build the final window_sensitivity JSON payload.
    Why: Stores diagnostics, sweep results, and freeze decision in one artifact.
    """
    prediction_window_validation = []
    if inter_purchase_gap_days:
        for window_days in FIXED_WINDOWS:
            gap_array = np.asarray(inter_purchase_gap_days, dtype=float)
            # Formula for this window: mean(gap_days <= window_days)
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
        "spend_cap_quantile": SPEND_CAP_QUANTILE,
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


def run_window_sensitivity(
    config_path: Path,
    feature_paths: list[Path],
    events_path: Path,
    output_json_path: Path,
    output_plot_path: Path,
) -> dict:
    """What: Run full window-sensitivity workflow and write JSON/plot artifacts.
    Why: Reusable in-process API for run_pipeline and CLI.
    """
    windows = FIXED_WINDOWS
    if set(windows) != ALLOWED_PREDICTION_WINDOWS:
        raise ValueError("--windows must be exactly 30,60,90 for strict architecture alignment.")

    inter_purchase_gap_days = _inter_purchase_gap_days(events_path)
    sweep_root = output_json_path.parent / "_window_sensitivity_train_runs"
    sensitivity_rows, feature_window_validation = _run_sensitivity_sweep(config_path, feature_paths, sweep_root)
    output = _build_output(
        feature_paths=feature_paths,
        events_path=events_path,
        inter_purchase_gap_days=inter_purchase_gap_days,
        sensitivity_rows=sensitivity_rows,
        feature_window_validation=feature_window_validation,
    )
    _write_window_sensitivity_artifact(output_json_path, output)
    print(f"Wrote sensitivity summary: {output_json_path}")
    _write_window_validation_dashboard(output, output_plot_path)
    return output


# ===== Entry Point =====
def main() -> None:
    """What: Orchestrate window sensitivity runs and write JSON/plot artifacts.
    Why: Automates model/window freeze decision under a fixed evaluation recipe.
    """
    parser = argparse.ArgumentParser(description="Run 30/60/90-day sensitivity for propensity models.")
    parser.add_argument("--config", required=True, help="Purchase propensity YAML config")
    args = parser.parse_args()

    cfg = load_yaml_defaults(args.config, "Engine config")
    output_root = Path("data")
    artifacts_dir = _cycle_artifacts_root(Path(args.config))
    panel_end_date = date.fromisoformat(str(cfg_required(cfg, "panel_end_date")))
    feature_paths = [
        output_root / "gold" / "feature_store" / "purchase_propensity" / "propensity_train_dataset" / f"as_of_date={snapshot.isoformat()}" / "propensity_train_dataset.parquet"
        for snapshot in generate_snapshot_dates(panel_end_date)
    ]
    events_path = output_root / "silver" / "transactions_line_items" / "transactions_line_items.parquet"
    missing_features = [path for path in feature_paths if not path.exists()]
    if missing_features:
        raise FileNotFoundError(f"Input path not found: {missing_features[0]}")
    if not events_path.exists():
        raise FileNotFoundError(f"Events path not found: {events_path}")
    run_window_sensitivity(
        config_path=Path(args.config),
        feature_paths=feature_paths,
        events_path=events_path,
        output_json_path=artifacts_dir / "offline_eval" / "window_sensitivity.json",
        output_plot_path=artifacts_dir / "offline_eval" / "window_validation_dashboard.png",
    )


if __name__ == "__main__":
    main()
