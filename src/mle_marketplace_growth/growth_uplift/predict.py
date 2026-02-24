"""Batch-score uplift from growth-uplift feature snapshots."""

import argparse
import csv
import pickle
from pathlib import Path


def _load_rows(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in input dataset: {input_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score uplift from feature snapshots.")
    parser.add_argument(
        "--input-csv",
        default="data/gold/feature_store/growth_uplift/user_features_asof/as_of_date=2011-12-09/user_features_asof.csv",
        help="Path to user feature snapshot CSV",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/growth_uplift/uplift_model.pkl",
        help="Path to trained uplift model bundle",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/growth_uplift/prediction_scores.csv",
        help="Path to output scored CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    model_path = Path(args.model_path)
    output_path = Path(args.output_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with model_path.open("rb") as file:
        model_bundle = pickle.load(file)

    vectorizer = model_bundle["vectorizer"]
    treated_model = model_bundle["treated_model"]
    control_model = model_bundle["control_model"]
    feature_columns = model_bundle["feature_columns"]

    rows = _load_rows(input_path)
    feature_rows = []
    for row in rows:
        features = {}
        for feature in feature_columns:
            if feature == "country":
                features[feature] = row[feature]
            else:
                features[feature] = float(row[feature])
        feature_rows.append(features)

    matrix = vectorizer.transform(feature_rows)
    treated_pred = treated_model.predict(matrix)
    control_pred = control_model.predict(matrix)
    uplift_pred = treated_pred - control_pred

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "as_of_date", "pred_treated", "pred_control", "uplift_score"])
        for row, pred_t, pred_c, uplift in zip(rows, treated_pred, control_pred, uplift_pred, strict=True):
            writer.writerow(
                [
                    row["user_id"],
                    row["as_of_date"],
                    round(float(pred_t), 6),
                    round(float(pred_c), 6),
                    round(float(uplift), 6),
                ]
            )

    print(f"Wrote prediction scores: {output_path}")


if __name__ == "__main__":
    main()
