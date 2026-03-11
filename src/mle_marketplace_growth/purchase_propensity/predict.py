"""Batch-score purchase propensity and expected value from feature snapshots."""

import argparse
import csv
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ===== Entry Point =====
def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Batch-score propensity and expected value from feature snapshots.")
    parser.add_argument("--input-path", default="data/gold/feature_store/purchase_propensity/user_features_asof/as_of_date=2011-12-09/user_features_asof.parquet", help="Path to user feature snapshot parquet")
    parser.add_argument("--model-path", default="artifacts/purchase_propensity/propensity_model.pkl", help="Path to trained propensity model bundle")
    parser.add_argument("--output-csv", default="artifacts/purchase_propensity/prediction_scores.csv", help="Path to output scored CSV")
    args = parser.parse_args()

    # ===== Input Checks =====
    input_path = Path(args.input_path)
    model_path = Path(args.model_path)
    output_path = Path(args.output_csv)
    if not input_path.exists(): raise FileNotFoundError(f"Input path not found: {input_path}")
    if not model_path.exists(): raise FileNotFoundError(f"Model not found: {model_path}")

    # ===== Load Trained Artifacts =====
    with model_path.open("rb") as file:
        model_bundle = pickle.load(file)

    vectorizer = model_bundle.get("vectorizer")
    encoded_feature_columns = model_bundle.get("encoded_feature_columns")
    propensity_model = model_bundle["propensity_model"]
    revenue_model = model_bundle["revenue_model"]
    revenue_fallback_value = float(model_bundle["revenue_fallback_value"])
    feature_columns = model_bundle["feature_columns"]
    spend_cap_value = float(model_bundle["spend_cap_value"])
    feature_lookback_days = int(model_bundle["feature_lookback_days"])
    prediction_window_days = int(model_bundle["prediction_window_days"])
    spend_feature = f"monetary_{feature_lookback_days}d"

    connection = duckdb.connect(database=":memory:")
    try:
        cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(input_path)])
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()
    if not rows: raise ValueError(f"No rows found in input dataset: {input_path}")

    # ===== Score Users =====
    feature_rows = []
    for row in rows:
        features = {}
        for feature in feature_columns:
            if feature == "country":
                features[feature] = row[feature]
            else:
                features[feature] = float(row[feature])
        features[spend_feature] = min(float(features[spend_feature]), spend_cap_value)
        feature_rows.append(features)

    # Intermediate features for scoring (not written to output directly).
    if encoded_feature_columns is not None:
        feature_frame = pd.DataFrame(feature_rows)
        encoded_frame = pd.get_dummies(feature_frame, columns=["country"], dtype=float)
        matrix = encoded_frame.reindex(columns=encoded_feature_columns, fill_value=0.0).to_numpy(dtype=float)
    elif vectorizer is not None:
        matrix = vectorizer.transform(feature_rows)
    else:
        raise ValueError("Model bundle is missing both encoded_feature_columns and vectorizer.")

    propensity_scores = propensity_model.predict_proba(matrix)[:, 1]
    if revenue_model is None:
        conditional_revenue_scores = np.full(len(feature_rows), max(0.0, revenue_fallback_value), dtype=float)
    else:
        conditional_revenue_scores = np.maximum(0.0, revenue_model.predict(matrix)).astype(float, copy=False)
    expected_value_scores = propensity_scores * conditional_revenue_scores

    # ===== Write Outputs =====
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "user_id",
                "as_of_date",
                "propensity_score",
                f"predicted_conditional_revenue_{prediction_window_days}d",
                "expected_value_score",
            ]
        )
        for row, propensity, conditional_revenue, expected_value in zip(rows, propensity_scores, conditional_revenue_scores, expected_value_scores, strict=True):
            writer.writerow(
                [
                    row["user_id"],
                    row["as_of_date"],
                    round(float(propensity), 6),
                    round(float(conditional_revenue), 6),
                    round(float(expected_value), 6),
                ]
            )

    print(f"Wrote prediction scores: {output_path}")


if __name__ == "__main__":
    main()
