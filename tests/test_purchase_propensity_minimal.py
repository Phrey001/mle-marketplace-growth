import csv
import tempfile
import unittest
from pathlib import Path

from mle_marketplace_growth.purchase_propensity.train import _load_training_rows, _policy_scores, _split_rows, _stable_ratio


class PurchasePropensityMinimalTests(unittest.TestCase):
    def _write_csv(self, fieldnames: list[str], rows: list[dict]) -> Path:
        temp_dir = Path(tempfile.mkdtemp())
        path = temp_dir / "input.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_window_overlap_test_chronological_split_order(self) -> None:
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "user_id": f"u{idx}",
                    "as_of_date": f"2020-{idx + 1:02d}-01",
                    "features": {
                        "recency_days": 5.0,
                        "frequency_30d": 1.0,
                        "frequency_90d": 2.0,
                        "monetary_30d": 10.0,
                        "monetary_90d": 20.0,
                        "avg_basket_value_90d": 10.0,
                    },
                    "purchase_label": 0.0,
                    "revenue_label": 0.0,
                }
            )
        train_rows, val_rows, test_rows, _ = _split_rows(rows)
        train_dates = {row["as_of_date"] for row in train_rows}
        val_dates = {row["as_of_date"] for row in val_rows}
        test_dates = {row["as_of_date"] for row in test_rows}
        self.assertEqual(len(train_dates), 10)
        self.assertEqual(len(val_dates), 1)
        self.assertEqual(len(test_dates), 1)
        self.assertLess(max(train_dates), next(iter(val_dates)))
        self.assertLess(next(iter(val_dates)), next(iter(test_dates)))

    def test_split_leakage_test_no_date_overlap_between_subsets(self) -> None:
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "user_id": f"user_{idx}",
                    "as_of_date": f"2021-{idx + 1:02d}-15",
                    "features": {
                        "recency_days": 3.0,
                        "frequency_30d": 1.0,
                        "frequency_90d": 1.0,
                        "monetary_30d": 5.0,
                        "monetary_90d": 7.0,
                        "avg_basket_value_90d": 7.0,
                    },
                    "purchase_label": 1.0 if idx % 2 else 0.0,
                    "revenue_label": 2.0,
                }
            )
        train_rows, val_rows, test_rows, _ = _split_rows(rows)
        train_dates = {row["as_of_date"] for row in train_rows}
        val_dates = {row["as_of_date"] for row in val_rows}
        test_dates = {row["as_of_date"] for row in test_rows}
        self.assertTrue(train_dates.isdisjoint(val_dates))
        self.assertTrue(train_dates.isdisjoint(test_dates))
        self.assertTrue(val_dates.isdisjoint(test_dates))

    def test_deterministic_seed_test_random_policy_score_is_stable(self) -> None:
        rows = [
            {
                "user_id": "u1",
                "as_of_date": "2021-01-01",
                "features": {"recency_days": 1.0, "frequency_90d": 2.0, "monetary_90d": 10.0},
            },
            {
                "user_id": "u2",
                "as_of_date": "2021-01-01",
                "features": {"recency_days": 2.0, "frequency_90d": 3.0, "monetary_90d": 20.0},
            },
        ]
        _, random_scores_first, _ = _policy_scores(rows, [0.1, 0.2], [100.0, 120.0], feature_lookback_days=90)
        _, random_scores_second, _ = _policy_scores(rows, [0.1, 0.2], [100.0, 120.0], feature_lookback_days=90)
        self.assertEqual(random_scores_first, random_scores_second)
        self.assertEqual(_stable_ratio("fixed_key"), _stable_ratio("fixed_key"))

    def test_feature_column_existence_test_missing_feature_column_raises(self) -> None:
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
        path = self._write_csv(fieldnames, rows)
        with self.assertRaises(KeyError):
            _load_training_rows(
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

