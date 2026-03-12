import csv
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from mle_marketplace_growth.purchase_propensity.helpers.data import _load_snapshot_rows, _split_df_rows_10_1_1
from mle_marketplace_growth.purchase_propensity.train import _policy_scores


class PurchasePropensityMinimalTests(unittest.TestCase):
    def _rows_with_monthly_dates(self, year: int, day: int, count: int = 12) -> list[dict]:
        return [{"user_id": f"u{idx}", "as_of_date": f"{year}-{idx + 1:02d}-{day:02d}"} for idx in range(count)]

    def _write_parquet(self, fieldnames: list[str], rows: list[dict]) -> Path:
        # Small helper for focused input fixtures per test.
        temp_dir = Path(tempfile.mkdtemp())
        csv_path = temp_dir / "input.csv"
        path = temp_dir / "input.parquet"
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute("CREATE OR REPLACE TABLE t AS SELECT * FROM read_csv_auto(?)", [str(csv_path)])
            connection.execute(f"COPY t TO '{str(path)}' (FORMAT PARQUET)")
        finally:
            connection.close()
        return path

    def test_window_overlap_test_chronological_split_order(self) -> None:
        # 12 monthly snapshots should split into 10/1/1 chronological buckets.
        train_df, val_df, test_df, _ = _split_df_rows_10_1_1(pd.DataFrame(self._rows_with_monthly_dates(2020, 1)))
        train_dates = set(train_df["as_of_date"].tolist())
        val_dates = set(val_df["as_of_date"].tolist())
        test_dates = set(test_df["as_of_date"].tolist())
        self.assertEqual(len(train_dates), 10)
        self.assertEqual(len(val_dates), 1)
        self.assertEqual(len(test_dates), 1)
        self.assertLess(max(train_dates), next(iter(val_dates)))
        self.assertLess(next(iter(val_dates)), next(iter(test_dates)))

    def test_split_leakage_test_no_date_overlap_between_subsets(self) -> None:
        # Train/validation/test snapshots must be disjoint.
        train_df, val_df, test_df, _ = _split_df_rows_10_1_1(pd.DataFrame(self._rows_with_monthly_dates(2021, 15)))
        train_dates = set(train_df["as_of_date"].tolist())
        val_dates = set(val_df["as_of_date"].tolist())
        test_dates = set(test_df["as_of_date"].tolist())
        self.assertTrue(train_dates.isdisjoint(val_dates))
        self.assertTrue(train_dates.isdisjoint(test_dates))
        self.assertTrue(val_dates.isdisjoint(test_dates))

    def test_deterministic_seed_test_random_policy_score_is_stable(self) -> None:
        # Random policy scores are seeded and should be reproducible.
        df = pd.DataFrame(
            [
                {"user_id": "u1", "as_of_date": "2021-01-01", "recency_days": 1.0, "frequency_90d": 2.0, "monetary_90d": 10.0},
                {"user_id": "u2", "as_of_date": "2021-01-01", "recency_days": 2.0, "frequency_90d": 3.0, "monetary_90d": 20.0},
            ]
        )
        _, random_scores_first, _ = _policy_scores(df, [0.1, 0.2], [100.0, 120.0], feature_lookback_days=90)
        _, random_scores_second, _ = _policy_scores(df, [0.1, 0.2], [100.0, 120.0], feature_lookback_days=90)
        self.assertEqual(random_scores_first, random_scores_second)

    def test_feature_column_existence_test_missing_feature_column_raises(self) -> None:
        # Missing required feature columns should fail fast at load time.
        fieldnames = [
            "user_id",
            "as_of_date",
            "country",
            "recency_days",
            "frequency_30d",
            "monetary_30d",
            "monetary_90d",
            "avg_basket_value_90d",
            "label_purchase_30d",
            "label_net_revenue_30d",
        ]
        # frequency_90d intentionally missing.
        rows = [
            {
                "user_id": "u1",
                "as_of_date": "2021-01-01",
                "country": "United Kingdom",
                "recency_days": "1",
                "frequency_30d": "1",
                "monetary_30d": "10",
                "monetary_90d": "20",
                "avg_basket_value_90d": "10",
                "label_purchase_30d": "1",
                "label_net_revenue_30d": "100",
            }
        ]
        path = self._write_parquet(fieldnames, rows)
        with self.assertRaises(ValueError):
            _load_snapshot_rows(
                path,
                feature_columns=[
                    "recency_days",
                    "frequency_30d",
                    "frequency_90d",
                    "monetary_30d",
                    "monetary_90d",
                    "avg_basket_value_90d",
                ],
                purchase_label_column="label_purchase_30d",
                revenue_label_column="label_net_revenue_30d",
            )


if __name__ == "__main__":
    unittest.main()
