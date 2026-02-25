import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

from mle_marketplace_growth.purchase_propensity import run_pipeline


class RunPipelineArgValidationTest(unittest.TestCase):
    def test_out_of_time_requires_multiple_snapshots(self) -> None:
        argv = [
            "run_pipeline.py",
            "--validation-mode",
            "out_of_time",
            "--train-as-of-dates",
            "2011-11-09",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(
                ValueError,
                "--validation-mode out_of_time requires at least 2 --train-as-of-dates",
            ):
                run_pipeline.main()

    def test_train_as_of_dates_cannot_be_empty(self) -> None:
        argv = [
            "run_pipeline.py",
            "--train-as-of-dates",
            ", ,",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "--train-as-of-dates must include at least one date"):
                run_pipeline.main()

    def test_config_requires_yaml_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = tmp_root / "pipeline_config.json"
            config_path.write_text('{"train_as_of_dates": "2011-11-09"}', encoding="utf-8")
            argv = ["run_pipeline.py", "--config", str(config_path)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "Config file must use .yaml or .yml"):
                    run_pipeline.main()

    def test_prediction_window_rejects_unsupported_main_pipeline_values(self) -> None:
        argv = [
            "run_pipeline.py",
            "--train-as-of-dates",
            "2011-11-09,2011-12-09",
            "--prediction-window-days",
            "60",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "implemented for 30-day labels only"):
                run_pipeline.main()

    def test_feature_lookback_window_rejects_unsupported_values(self) -> None:
        argv = [
            "run_pipeline.py",
            "--train-as-of-dates",
            "2011-11-09,2011-12-09",
            "--feature-lookback-days",
            "30",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "--feature-lookback-days must be one of"):
                run_pipeline.main()

    def test_feature_lookback_window_rejects_not_yet_wired_values(self) -> None:
        argv = [
            "run_pipeline.py",
            "--train-as-of-dates",
            "2011-11-09,2011-12-09",
            "--feature-lookback-days",
            "120",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "implemented for 90-day feature lookback only"):
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
            config_path.write_text("train_as_of_dates: '2011-11-09'\n", encoding="utf-8")

            argv = [
                "run_pipeline.py",
                "--config",
                str(config_path),
            ]
            with patch("sys.argv", argv):
                with patch(
                    "mle_marketplace_growth.purchase_propensity.run_pipeline._read_config_file",
                    return_value={
                        "train_as_of_dates": "2011-11-09",
                        "score_as_of_date": "2011-11-09",
                    },
                ):
                    with patch("mle_marketplace_growth.purchase_propensity.run_pipeline._build_snapshot") as mock_build:
                        with patch("mle_marketplace_growth.purchase_propensity.run_pipeline._run") as mock_run:
                            with patch("mle_marketplace_growth.purchase_propensity.run_pipeline.run_validation", return_value=(True, {"checks": []})):
                                with patch("mle_marketplace_growth.purchase_propensity.run_pipeline.write_interpretation"):
                                    run_pipeline.main()
            self.assertTrue(mock_build.called)
            self.assertTrue(mock_run.called)


if __name__ == "__main__":
    unittest.main()
