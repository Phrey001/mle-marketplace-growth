"""Batch-score purchase propensity and expected value from feature snapshots."""

import argparse
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mle_marketplace_growth.helpers import cfg_required, load_yaml_defaults
from mle_marketplace_growth.purchase_propensity.helpers.artifacts import (
    _offline_eval_paths,
    _serving_prediction_scores_path,
    _write_batch_prediction_scores_artifact,
)
from mle_marketplace_growth.purchase_propensity.helpers.data import _read_parquet_panel


# ===== Entry Point =====
def main() -> None:
    """What: Load a trained artifact and score one feature snapshot parquet.
    Why: Supports optional standalone batch scoring outside the full pipeline run.
    """
    # ===== CLI Args =====
    parser = argparse.ArgumentParser(description="Batch-score propensity and expected value from feature snapshots.")
    parser.add_argument("--config", required=True, help="Purchase propensity YAML config")
    args = parser.parse_args()

    # ===== Load Config =====
    cfg = load_yaml_defaults(args.config, "Engine config")
    output_root = Path(str(cfg_required(cfg, "output_root")))
    artifacts_dir = Path(str(cfg_required(cfg, "artifacts_dir")))
    panel_end_date = date.fromisoformat(str(cfg_required(cfg, "panel_end_date")))
    panel_end_iso = panel_end_date.isoformat()
    default_input_path = (
        output_root
        / "gold"
        / "feature_store"
        / "purchase_propensity"
        / "user_features_asof"
        / f"as_of_date={panel_end_iso}"
        / "user_features_asof.parquet"
    )
    default_model_path = _offline_eval_paths(artifacts_dir).model_path
    default_output_csv = _serving_prediction_scores_path(artifacts_dir, panel_end_date)

    # ===== Validate Inputs =====
    input_path = default_input_path
    model_path = default_model_path
    output_path = default_output_csv
    if not input_path.exists(): raise FileNotFoundError(f"Input path not found: {input_path}")
    if not model_path.exists(): raise FileNotFoundError(f"Model not found: {model_path}")

    # ===== Load Inputs =====
    with model_path.open("rb") as file:
        model_bundle = pickle.load(file)

    # Resolve trained objects + run metadata from model artifact.
    encoded_feature_columns = model_bundle.get("encoded_feature_columns")
    if encoded_feature_columns is None:
        raise ValueError("Model bundle missing 'encoded_feature_columns'. Retrain with current train.py pipeline.")
    propensity_model = model_bundle["propensity_model"]
    revenue_model = model_bundle["revenue_model"]
    revenue_fallback_value = float(model_bundle["revenue_fallback_value"])
    feature_columns = model_bundle["feature_columns"]
    spend_cap_value = float(model_bundle["spend_cap_value"])
    feature_lookback_days = int(model_bundle["feature_lookback_days"])
    prediction_window_days = int(model_bundle["prediction_window_days"])
    spend_feature = f"monetary_{feature_lookback_days}d"

    snapshot_df = _read_parquet_panel(input_path, allow_empty=True)
    if snapshot_df.empty:
        raise ValueError(f"No rows found in input dataset: {input_path}")

    # ===== Score Users =====
    # 1) Build model input columns from the scored snapshot.
    # Keep the exact train-time feature set plus `country` for one-hot encoding.
    feature_input_columns = list(feature_columns)
    if "country" not in feature_input_columns:
        feature_input_columns.append("country")
    missing_columns = [column for column in feature_input_columns if column not in snapshot_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in input snapshot: {missing_columns}")
    feature_df = snapshot_df[feature_input_columns].copy()

    # 2) Match train-time dtypes and clip long-window spend with the frozen cap.
    feature_df["country"] = feature_df["country"].astype(str)
    numeric_feature_columns = [column for column in feature_columns if column != "country"]
    feature_df[numeric_feature_columns] = feature_df[numeric_feature_columns].astype(float)
    if spend_feature not in feature_df.columns:
        raise ValueError(f"Spend feature not found in snapshot: {spend_feature}")
    feature_df[spend_feature] = feature_df[spend_feature].clip(upper=spend_cap_value)

    # 3) Recreate encoded feature matrix using train-time column order.
    # Any unseen category becomes 0 via reindex(fill_value=0.0).
    encoded_df = pd.get_dummies(feature_df, columns=["country"], dtype=float)
    matrix = encoded_df.reindex(columns=encoded_feature_columns, fill_value=0.0).to_numpy(dtype=float)

    # 4) Score propensity p(purchase|x), then conditional revenue E(revenue|purchase=1,x).
    # 5) Compute expected value score = propensity * conditional revenue.
    propensity_scores = propensity_model.predict_proba(matrix)[:, 1]
    if revenue_model is None:
        conditional_revenue_scores = np.full(len(feature_df), max(0.0, revenue_fallback_value), dtype=float)
    else:
        conditional_revenue_scores = np.maximum(0.0, revenue_model.predict(matrix)).astype(float, copy=False)
    expected_value_scores = propensity_scores * conditional_revenue_scores

    # ===== Write Outputs =====
    _write_batch_prediction_scores_artifact(
        output_path,
        snapshot_df,
        propensity_scores,
        conditional_revenue_scores,
        expected_value_scores,
        prediction_window_days,
    )

    print(f"Wrote prediction scores: {output_path}")


if __name__ == "__main__":
    main()
