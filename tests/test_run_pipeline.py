import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

from mle_marketplace_growth.purchase_propensity import run_pipeline


class RunPipelineArgValidationTest(unittest.TestCase):
    STRICT_START_DATE = "2010-12-09"
    STRICT_END_DATE = "2011-11-09"

    def test_strict_mode_requires_exactly_twelve_snapshots(self) -> None:
        argv = [
            "run_pipeline.py",
            "--train-start-date",
            "2011-01-09",
            "--train-end-date",
            "2011-09-09",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(
                ValueError,
                "Strict architecture split requires exactly 12 monthly snapshots",
            ):
                run_pipeline.main()

    def test_out_of_time_10_1_1_requires_exactly_twelve_snapshots(self) -> None:
        argv = [
            "run_pipeline.py",
            "--train-start-date",
            "2011-03-09",
            "--train-end-date",
            "2011-11-09",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(
                ValueError,
                "Strict architecture split requires exactly 12 monthly snapshots",
            ):
                run_pipeline.main()

    def test_train_start_and_end_dates_are_required(self) -> None:
        argv = [
            "run_pipeline.py",
            "--train-start-date",
            self.STRICT_START_DATE,
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "Both --train-start-date and --train-end-date are required"):
                run_pipeline.main()

    def test_config_requires_yaml_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = tmp_root / "pipeline_config.json"
            config_path.write_text('{"train_as_of_dates": "2011-11-09"}', encoding="utf-8")
            argv = ["run_pipeline.py", "--config", str(config_path)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "Engine config file must use .yaml or .yml"):
                    run_pipeline.main()

    def test_prediction_window_rejects_unsupported_main_pipeline_values(self) -> None:
        argv = [
            "run_pipeline.py",
            "--train-start-date",
            self.STRICT_START_DATE,
            "--train-end-date",
            self.STRICT_END_DATE,
            "--prediction-window-days",
            "45",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "--prediction-window-days must be one of"):
                run_pipeline.main()

    def test_feature_lookback_window_rejects_unsupported_values(self) -> None:
        argv = [
            "run_pipeline.py",
            "--train-start-date",
            self.STRICT_START_DATE,
            "--train-end-date",
            self.STRICT_END_DATE,
            "--feature-lookback-days",
            "30",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "--feature-lookback-days must be one of"):
                run_pipeline.main()

    def test_feature_lookback_window_rejects_not_yet_wired_values(self) -> None:
        argv = [
            "run_pipeline.py",
            "--train-start-date",
            self.STRICT_START_DATE,
            "--train-end-date",
            self.STRICT_END_DATE,
            "--feature-lookback-days",
            "150",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "--feature-lookback-days must be one of"):
                run_pipeline.main()

    def test_merge_train_datasets_combines_rows_with_one_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            train_a = tmp_root / "train_a.csv"
            train_b = tmp_root / "train_b.csv"
            merged = tmp_root / "merged.csv"

            fieldnames = ["user_id", "as_of_date", "label_purchase_30d"]
            with train_a.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"user_id": "u1", "as_of_date": "2011-10-09", "label_purchase_30d": "1"})
            with train_b.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"user_id": "u2", "as_of_date": "2011-11-09", "label_purchase_30d": "0"})

            run_pipeline._merge_train_datasets([train_a, train_b], merged)

            with merged.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["user_id"], "u1")
            self.assertEqual(rows[1]["user_id"], "u2")

    def test_config_file_executes_pipeline_without_long_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = tmp_root / "pipeline_config.yaml"
            config_path.write_text(
                "train_start_date: '2010-12-09'\n"
                "train_end_date: '2011-11-09'\n"
                "window_selection_mode: fixed\n",
                encoding="utf-8",
            )

            argv = [
                "run_pipeline.py",
                "--config",
                str(config_path),
            ]
            with patch("sys.argv", argv):
                with patch("mle_marketplace_growth.purchase_propensity.run_pipeline._run_module") as mock_run:
                    with patch("mle_marketplace_growth.purchase_propensity.run_pipeline.run_validation", return_value=(True, {"checks": []})):
                        with patch("mle_marketplace_growth.purchase_propensity.run_pipeline.write_interpretation"):
                            run_pipeline.main()
            self.assertTrue(mock_run.called)


if __name__ == "__main__":
    unittest.main()
