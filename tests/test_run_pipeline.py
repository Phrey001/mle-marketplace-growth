import unittest
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
            config_path = self._write_yaml(
                tmp_root,
                "pipeline_config.yaml",
                "panel_end_date: '2011-11-09'\nprediction_window_days: 45\n",
            )
            argv = ["run_pipeline.py", "--config", str(config_path)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "--prediction-window-days must be one of"):
                    run_pipeline.main()

    def test_feature_lookback_window_rejects_unsupported_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = self._write_yaml(
                tmp_root,
                "pipeline_config.yaml",
                "panel_end_date: '2011-11-09'\nfeature_lookback_days: 30\n",
            )
            argv = ["run_pipeline.py", "--config", str(config_path)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "--feature-lookback-days must be one of"):
                    run_pipeline.main()

    def test_feature_lookback_window_rejects_not_yet_wired_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = self._write_yaml(
                tmp_root,
                "pipeline_config.yaml",
                "panel_end_date: '2011-11-09'\nfeature_lookback_days: 150\n",
            )
            argv = ["run_pipeline.py", "--config", str(config_path)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "--feature-lookback-days must be one of"):
                    run_pipeline.main()

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
                "window_selection_mode: fixed\n"
                "force_propensity_model: logistic_regression\n",
                encoding="utf-8",
            )
            for as_of_date in run_pipeline._generate_snapshot_dates(run_pipeline.date.fromisoformat(self.PANEL_END_DATE)):
                dataset_path = gold_root / f"as_of_date={as_of_date}" / "propensity_train_dataset.parquet"
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                dataset_path.touch()

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
