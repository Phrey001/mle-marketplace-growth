"""Train purchase propensity scoring models and offline policy backtest."""

import argparse
import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mle_marketplace_growth.helpers import cfg_required, generate_snapshot_dates, load_yaml_defaults
from mle_marketplace_growth.purchase_propensity.constants import (
    ALLOWED_FEATURE_LOOKBACK_WINDOWS,
    ALLOWED_PREDICTION_WINDOWS,
    SPEND_CAP_QUANTILE,
)
from mle_marketplace_growth.purchase_propensity.helpers.artifacts import (
    _build_train_metrics_payload,
    _dump_model_artifact,
    _offline_eval_paths,
    _write_metrics_artifact,
    _write_predictions_csv,
)
from mle_marketplace_growth.purchase_propensity.helpers.data import (
    _load_snapshot_rows,
    _quantile,
    _split_df_rows_10_1_1,
)
from mle_marketplace_growth.purchase_propensity.helpers.metrics import _propensity_quality_metrics
from mle_marketplace_growth.purchase_propensity.helpers.models import (
    _fit_test_conditional_revenue_model_wrapper,
    _fit_test_propensity_model_wrapper,
    _fit_validation_conditional_revenue_model_wrapper,
    _fit_validation_propensity_model_wrapper,
)

def _stable_ratio(key: str) -> float:
    """What: Deterministically map a string key to a pseudo-random ratio in [0, 1).
    Why: Keeps random-baseline ranking reproducible across runs and tests.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _apply_spend_cap(df: pd.DataFrame, spend_feature: str, spend_cap_value: float) -> None:
    """What: Cap one spend feature column in-place.
    Why: Limits outlier skew before model fitting and scoring.
    """
    df[spend_feature] = np.minimum(df[spend_feature].to_numpy(dtype=float), spend_cap_value)


def _fit_feature_encoding(train_feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """What: One-hot encode train features and return encoded frame plus column order.
    Why: Freezes the train schema so non-train splits can be aligned without leakage.
    """
    train_encoded_df = pd.get_dummies(train_feature_df, columns=["country"], dtype=float)
    return train_encoded_df, train_encoded_df.columns.tolist()


def _apply_feature_encoding(feature_df: pd.DataFrame, encoded_columns: list[str]) -> pd.DataFrame:
    """What: One-hot encode a split and align columns to the fitted train schema.
    Why: Keeps feature dimensions/order identical across train, validation, and test.
    """
    encoded_df = pd.get_dummies(feature_df, columns=["country"], dtype=float)
    return encoded_df.reindex(columns=encoded_columns, fill_value=0.0)


def _policy_scores(
    df: pd.DataFrame,
    propensity_scores: np.ndarray,
    predicted_conditional_revenue: np.ndarray,
    feature_lookback_days: int,
) -> tuple[list[float], list[float], list[float]]:
    """What: Build expected-value, random, and RFM policy scores per row.
    Why: Enables side-by-side offline policy comparison on the same population.
    """
    propensity_scores = np.asarray(propensity_scores, dtype=float)
    predicted_conditional_revenue = np.asarray(predicted_conditional_revenue, dtype=float)

    # A) Expected-value policy: propensity * predicted conditional revenue.
    expected_value_scores = (propensity_scores * predicted_conditional_revenue).tolist()
    # B) Random baseline policy: deterministic pseudo-random ranking per user/date.
    random_scores = [
        1.0 - _stable_ratio(f"{user_id}|{as_of_date}|policy_random")
        for user_id, as_of_date in zip(df["user_id"], df["as_of_date"], strict=True)
    ]
    # C) RFM heuristic policy: weighted recency + frequency + monetary blend.
    freq_feature, monetary_feature = f"frequency_{feature_lookback_days}d", f"monetary_{feature_lookback_days}d"
    rfm_scores = (
        (1.0 / (1.0 + df["recency_days"].to_numpy(dtype=float)))
        + 0.5 * df[freq_feature].to_numpy(dtype=float)
        + 0.01 * df[monetary_feature].to_numpy(dtype=float)
    ).tolist()
    return expected_value_scores, random_scores, rfm_scores


def run_training(
    config_path: Path,
    input_paths: list[Path] | None = None,
    output_dir: Path | None = None,
    prediction_window_days: int | None = None,
    feature_lookback_days: int | None = None,
    force_propensity_model: str | None = None,
) -> None:
    """What: Train propensity + conditional revenue models and write offline artifacts.
    Why: Provides one in-process entrypoint shared by CLI and orchestrators.
    """
    # 0) Load config and resolve effective runtime settings.
    cfg = load_yaml_defaults(str(config_path), "Engine config")

    panel_end_date = date.fromisoformat(str(cfg_required(cfg, "panel_end_date")))
    output_root = Path(str(cfg_required(cfg, "output_root")))
    artifacts_dir = Path(str(cfg_required(cfg, "artifacts_dir")))
    default_offline_paths = _offline_eval_paths(artifacts_dir)
    prediction_window_days = int(prediction_window_days) if prediction_window_days is not None else int(cfg_required(cfg, "prediction_window_days"))
    feature_lookback_days = int(feature_lookback_days) if feature_lookback_days is not None else int(cfg_required(cfg, "feature_lookback_days"))
    force_propensity_model = force_propensity_model or cfg.get("force_propensity_model", None)
    if not force_propensity_model:
        raise ValueError("--force-propensity-model is required for train.py (set in config fixed mode or pass CLI override)")
    output_dir = output_dir or default_offline_paths.root

    # 1) Validate inputs and window settings.
    if prediction_window_days not in ALLOWED_PREDICTION_WINDOWS: raise ValueError("--prediction-window-days must be one of: 30, 60, 90")
    if feature_lookback_days not in ALLOWED_FEATURE_LOOKBACK_WINDOWS: raise ValueError("--feature-lookback-days must be one of: 60, 90, 120")

    # 2) Resolve input snapshots for the strict 12-month panel.
    if input_paths is None:
        input_paths = [
            output_root / "gold" / "feature_store" / "purchase_propensity" / "propensity_train_dataset" / f"as_of_date={as_of_date}" / "propensity_train_dataset.parquet"
            for as_of_date in [snapshot.isoformat() for snapshot in generate_snapshot_dates(panel_end_date)]
        ]
    missing_inputs = [path for path in input_paths if not path.exists()]  # collect paths missing on disk if any
    if missing_inputs: raise FileNotFoundError(f"Input path not found: {missing_inputs[0]}")

    # 3) Define feature + label columns for this run.
    feature_columns = [
        # always include recency and short-term signals
        "recency_days",
        "frequency_30d",
        "monetary_30d",
        # dependent on lookback days
        f"frequency_{feature_lookback_days}d",
        f"monetary_{feature_lookback_days}d",
        f"avg_basket_value_{feature_lookback_days}d",
    ]
    spend_feature = f"monetary_{feature_lookback_days}d"  # feature already exist in feature_columns; for mutation in-place later
    purchase_label_column = f"label_purchase_{prediction_window_days}d"  # selected-horizon purchase label; used for loading and CSV schema
    revenue_label_column = f"label_net_revenue_{prediction_window_days}d"  # selected-horizon revenue label; used for loading revenue targets and CSV schema

    # 4) Load DataFrame, split into 10/1/1, and cap long-window spend.
    data_df = _load_snapshot_rows(
        input_paths,
        feature_columns=feature_columns,
        purchase_label_column=purchase_label_column,
        revenue_label_column=revenue_label_column,
    )
    train_df, validation_df, test_df, split_description = _split_df_rows_10_1_1(data_df)
    if train_df.empty or validation_df.empty or test_df.empty: raise ValueError("Train/validation/test split produced an empty subset.")

    train_purchase_labels = train_df["purchase_label"].to_numpy(dtype=int)
    if np.unique(train_purchase_labels).size < 2: raise ValueError("Training labels contain only one class. Try an earlier as_of_date.")

    spend_cap_value = _quantile(train_df[spend_feature].to_numpy(dtype=float), SPEND_CAP_QUANTILE)
    _apply_spend_cap(train_df, spend_feature, spend_cap_value)
    _apply_spend_cap(validation_df, spend_feature, spend_cap_value)
    _apply_spend_cap(test_df, spend_feature, spend_cap_value)

    # Feature preparation only (model training starts after this block).
    # Build raw feature tables (numeric + country) for this split.
    train_feature_df = train_df[[*feature_columns, "country"]].copy()
    validation_feature_df = validation_df[[*feature_columns, "country"]].copy()
    train_encoded_df, encoded_feature_columns = _fit_feature_encoding(train_feature_df)  # Fit one-hot schema on train only to avoid leakage.
    validation_encoded_df = _apply_feature_encoding(validation_feature_df, encoded_feature_columns)  # Apply same schema to non-train split, then convert to model matrices.
    train_matrix = train_encoded_df.to_numpy(dtype=float)
    validation_matrix = validation_encoded_df.to_numpy(dtype=float)
    validation_purchase_labels = validation_df["purchase_label"].to_numpy(dtype=int)
    train_conditional_revenue_labels = train_df["revenue_label"].to_numpy(dtype=float)
    validation_conditional_revenue_labels = validation_df["revenue_label"].to_numpy(dtype=float)

    # 5) Propensity model: train only the forced model for deterministic execution.
    selected_result = _fit_validation_propensity_model_wrapper(
        train_matrix,
        train_purchase_labels,
        validation_matrix,
        validation_purchase_labels,
        model_name=force_propensity_model,
        bins=10,
    )
    selected_model_name = str(selected_result["model_name"])
    validation_propensity_scores = selected_result["propensity_scores"]
    selected_roc_auc = float(selected_result["roc_auc"])
    selected_average_precision = float(selected_result["average_precision"])

    validation_quality_metrics = {
        "top_decile_lift": float(selected_result["top_decile_lift"]),
        "top_decile_positive_rate": float(selected_result["top_decile_positive_rate"]),
        "base_positive_rate": float(selected_result["base_positive_rate"]),
        "brier_score": float(selected_result["brier_score"]),
        "ece_10_bin": float(selected_result["ece_10_bin"]),
    }

    # 6) Revenue model: request XGB on validation, then freeze the effective path (XGB or fallback constant).
    # 6a) Fit/score conditional revenue on validation rows and freeze the effective model choice.
    _, frozen_revenue_model_name, _, validation_revenue_predictions, revenue_validation_quality = _fit_validation_conditional_revenue_model_wrapper(
        train_matrix,
        train_purchase_labels,
        train_conditional_revenue_labels,
        validation_matrix,
        validation_purchase_labels,
        validation_conditional_revenue_labels,
    )

    # 7) Final refit: selected propensity + revenue models on train+validation and score test.
    train_val_df = pd.concat([train_df, validation_df], ignore_index=True)
    train_val_purchase_labels = train_val_df["purchase_label"].to_numpy(dtype=int)
    train_val_conditional_revenue_labels = train_val_df["revenue_label"].to_numpy(dtype=float)
    train_val_feature_df = train_val_df[[*feature_columns, "country"]].copy()
    test_feature_df = test_df[[*feature_columns, "country"]].copy()
    train_val_encoded_df, train_val_encoded_feature_columns = _fit_feature_encoding(train_val_feature_df)
    test_encoded_df = _apply_feature_encoding(test_feature_df, train_val_encoded_feature_columns)
    train_val_matrix = train_val_encoded_df.to_numpy(dtype=float)
    test_matrix = test_encoded_df.to_numpy(dtype=float)

    selected_model, test_propensity_scores = _fit_test_propensity_model_wrapper(
        train_val_matrix,
        train_val_purchase_labels,
        test_matrix,
        selected_model_name,
    )
    revenue_model, revenue_model_name, revenue_fallback_value, test_predicted_conditional_revenue, revenue_test_quality = _fit_test_conditional_revenue_model_wrapper(
        train_val_matrix,
        train_val_purchase_labels,
        train_val_conditional_revenue_labels,
        test_matrix,
        test_df["purchase_label"].to_numpy(dtype=int),
        test_df["revenue_label"].to_numpy(dtype=float),
        frozen_revenue_model_name=frozen_revenue_model_name,
    )

    test_purchase_labels = test_df["purchase_label"].to_numpy(dtype=int)
    test_quality_metrics = _propensity_quality_metrics(test_purchase_labels, test_propensity_scores, bins=10)

    # 8) Build policy scores for validation/test slices.
    test_expected_value_scores, test_random_scores, test_rfm_scores = _policy_scores(test_df, test_propensity_scores, test_predicted_conditional_revenue, feature_lookback_days)
    validation_expected_value_scores, validation_random_scores, validation_rfm_scores = _policy_scores(validation_df, validation_propensity_scores, validation_revenue_predictions, feature_lookback_days)

    # 9) Persist model, metrics, and scored outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = default_offline_paths if output_dir == default_offline_paths.root else None
    model_path = (paths.model_path if paths else output_dir / "propensity_model.pkl")
    metrics_path = (paths.metrics_path if paths else output_dir / "train_metrics.json")
    validation_scores_path = (paths.validation_predictions_path if paths else output_dir / "validation_predictions.csv")
    test_scores_path = (paths.test_predictions_path if paths else output_dir / "test_predictions.csv")

    _dump_model_artifact(
        model_path,
        selected_model,
        selected_model_name,
        revenue_model,
        revenue_model_name,
        revenue_fallback_value,
        feature_columns,
        train_val_encoded_feature_columns,
        spend_cap_value,
        prediction_window_days,
        feature_lookback_days,
    )

    metrics_payload = _build_train_metrics_payload(
        input_paths,
        purchase_label_column,
        revenue_label_column,
        prediction_window_days,
        feature_lookback_days,
        split_description,
        spend_cap_value,
        revenue_model_name,
        revenue_fallback_value,
        len(train_df),
        len(validation_df),
        len(test_df),
        selected_model_name,
        selected_roc_auc,
        selected_average_precision,
        validation_quality_metrics,
        test_purchase_labels,
        test_propensity_scores,
        test_quality_metrics,
        revenue_validation_quality,
        revenue_test_quality,
    )
    _write_metrics_artifact(metrics_path, metrics_payload)

    _write_predictions_csv(
        validation_scores_path,
        validation_df,
        purchase_label_column,
        revenue_label_column,
        prediction_window_days,
        validation_propensity_scores,
        validation_revenue_predictions,
        validation_expected_value_scores,
        validation_random_scores,
        validation_rfm_scores,
    )
    _write_predictions_csv(
        test_scores_path,
        test_df,
        purchase_label_column,
        revenue_label_column,
        prediction_window_days,
        test_propensity_scores,
        test_predicted_conditional_revenue,
        test_expected_value_scores,
        test_random_scores,
        test_rfm_scores,
    )

    print(f"Wrote model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote validation predictions: {validation_scores_path}")
    print(f"Wrote test predictions: {test_scores_path}")
    print(f"Selected model: {selected_model_name}")


def main() -> None:
    """What: Train propensity and conditional revenue models, then write artifacts.
    Why: Produces deterministic offline backtest outputs for policy evaluation.
    """
    # ===== CLI Args =====
    parser = argparse.ArgumentParser(description="Train propensity scoring models and backtest targeting policies.")
    parser.add_argument("--config", required=True, help="Purchase propensity YAML config")
    args = parser.parse_args()

    run_training(
        config_path=Path(args.config),
    )


if __name__ == "__main__":
    main()
