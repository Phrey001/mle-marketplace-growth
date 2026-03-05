import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

from mle_marketplace_growth.purchase_propensity import run_pipeline


class RunPipelineArgValidationTest(unittest.TestCase):
    PANEL_END_DATE = "2011-11-09"

    def _write_yaml(self, tmp_root: Path, name: str, content: str) -> Path:
        path = tmp_root / name
        path.write_text(content, encoding="utf-8")
        return path

    # ===== Argument + Config Validation =====
    def test_panel_end_date_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = self._write_yaml(tmp_root, "pipeline_config.yaml", "window_selection_mode: fixed\n")
            argv = ["run_pipeline.py", "--config", str(config_path)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "--panel-end-date is required"):
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = self._write_yaml(tmp_root, "pipeline_config.yaml", "")
            argv = [
                "run_pipeline.py",
                "--config",
                str(config_path),
                "--panel-end-date",
                self.PANEL_END_DATE,
                "--prediction-window-days",
                "45",
            ]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "--prediction-window-days must be one of"):
                    run_pipeline.main()

    def test_feature_lookback_window_rejects_unsupported_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = self._write_yaml(tmp_root, "pipeline_config.yaml", "")
            argv = [
                "run_pipeline.py",
                "--config",
                str(config_path),
                "--panel-end-date",
                self.PANEL_END_DATE,
                "--feature-lookback-days",
                "30",
            ]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "--feature-lookback-days must be one of"):
                    run_pipeline.main()

    def test_feature_lookback_window_rejects_not_yet_wired_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = self._write_yaml(tmp_root, "pipeline_config.yaml", "")
            argv = [
                "run_pipeline.py",
                "--config",
                str(config_path),
                "--panel-end-date",
                self.PANEL_END_DATE,
                "--feature-lookback-days",
                "150",
            ]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "--feature-lookback-days must be one of"):
                    run_pipeline.main()

    # ===== Data Merge Helper =====
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

    # ===== Config-Driven Orchestration =====
    def test_config_file_executes_pipeline_without_long_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            output_root = tmp_root / "data"
            gold_root = output_root / "gold" / "feature_store" / "purchase_propensity" / "propensity_train_dataset"
            config_path = tmp_root / "pipeline_config.yaml"
            config_path.write_text(
                "output_root: " + str(output_root) + "\n"
                "panel_end_date: '2011-11-09'\n"
                "window_selection_mode: fixed\n",
                encoding="utf-8",
            )
            for as_of_date in run_pipeline._generate_snapshot_dates(run_pipeline.date.fromisoformat(self.PANEL_END_DATE)):
                dataset_path = gold_root / f"as_of_date={as_of_date}" / "propensity_train_dataset.csv"
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                dataset_path.write_text("user_id,as_of_date,label_purchase_30d\nu1,2011-11-09,0\n", encoding="utf-8")

            argv = [
                "run_pipeline.py",
                "--config",
                str(config_path),
            ]
            with patch("sys.argv", argv):
                with patch("mle_marketplace_growth.purchase_propensity.run_pipeline._run_module") as mock_run:
                    with patch("mle_marketplace_growth.purchase_propensity.run_pipeline._merge_train_datasets"):
                        with patch("mle_marketplace_growth.purchase_propensity.run_pipeline.run_validation", return_value=(True, {"checks": []})):
                            with patch("mle_marketplace_growth.purchase_propensity.run_pipeline.write_interpretation"):
                                run_pipeline.main()
            self.assertTrue(mock_run.called)


if __name__ == "__main__":
    unittest.main()
