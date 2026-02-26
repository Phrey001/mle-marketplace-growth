import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mle_marketplace_growth.recommender.train import (
    _load_split_rows,
    _train_two_tower,
    _user_eval_pool,
    _validate_split_chronology,
)


class RecommenderMinimalTests(unittest.TestCase):
    def _write_rows(self, rows: list[dict]) -> Path:
        temp_dir = Path(tempfile.mkdtemp())
        path = temp_dir / "splits.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_window_overlap_test_split_chronology_violation(self) -> None:
        rows = [
            {"user_id": "u1", "item_id": "i1", "split": "train", "event_ts": "2024-01-03 00:00:00"},
            {"user_id": "u1", "item_id": "i2", "split": "val", "event_ts": "2024-01-02 00:00:00"},
            {"user_id": "u1", "item_id": "i3", "split": "test", "event_ts": "2024-01-04 00:00:00"},
        ]
        with self.assertRaisesRegex(ValueError, "Split chronology violation"):
            _validate_split_chronology(rows)

    def test_split_leakage_test_eval_pool_excludes_train_items(self) -> None:
        train_items = {"i1", "i2"}
        gt_items = {"i3"}
        item_to_idx = {"i1": 0, "i2": 1, "i3": 2, "i4": 3}
        pool = _user_eval_pool(train_items, gt_items, item_to_idx)
        self.assertIsNotNone(pool)
        candidates, _ = pool
        self.assertNotIn(0, candidates)
        self.assertNotIn(1, candidates)

    def test_deterministic_seed_test_two_tower_same_result(self) -> None:
        train = {"u1": {"i1", "i2"}, "u2": {"i2", "i3"}}
        user_to_idx = {"u1": 0, "u2": 1}
        item_to_idx = {"i1": 0, "i2": 1, "i3": 2}
        first_user, first_item = _train_two_tower(
            train=train,
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx,
            embedding_dim=8,
            epochs=1,
            learning_rate=0.05,
            negative_samples=2,
            l2_reg=1e-4,
        )
        second_user, second_item = _train_two_tower(
            train=train,
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx,
            embedding_dim=8,
            epochs=1,
            learning_rate=0.05,
            negative_samples=2,
            l2_reg=1e-4,
        )
        self.assertTrue(np.allclose(first_user, second_user))
        self.assertTrue(np.allclose(first_item, second_item))

    def test_feature_column_existence_test_required_columns(self) -> None:
        rows = [
            {"user_id": "u1", "item_id": "i1", "split": "train"},
        ]
        path = self._write_rows(rows)
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            _load_split_rows(path)


if __name__ == "__main__":
    unittest.main()
