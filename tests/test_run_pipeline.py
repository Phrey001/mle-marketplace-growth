import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from mle_marketplace_growth.helpers import generate_snapshot_dates
from mle_marketplace_growth.purchase_propensity import run_pipeline


class RunPipelineArgValidationTest(unittest.TestCase):
    PANEL_END_DATE = "2011-11-09"

    def _base_required_config(self, output_root: Path | None = None) -> str:
        root = output_root or Path("data")
        return (
            f"output_root: {root}\n"
            "artifacts_dir: artifacts/purchase_propensity/test_cycle\n"
            "panel_end_date: '2011-11-09'\n"
            "window_selection_mode: fixed\n"
            "force_propensity_model: logistic_regression\n"
            "prediction_window_days: 30\n"
            "feature_lookback_days: 90\n"
            "budget: 5000.0\n"
            "cost_per_user: 5.0\n"
        )

    def _write_yaml(self, tmp_root: Path, name: str, content: str) -> Path:
        path = tmp_root / name
        path.write_text(content, encoding="utf-8")
        return path

    def _run_with_config(self, config_path: Path) -> None:
        with patch("sys.argv", ["run_pipeline.py", "--config", str(config_path)]):
            run_pipeline.main()

    # ===== Argument + Config Validation =====
    def test_panel_end_date_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = self._write_yaml(
                tmp_root,
                "pipeline_config.yaml",
                self._base_required_config().replace("panel_end_date: '2011-11-09'\n", ""),
            )
            with self.assertRaisesRegex(ValueError, "Missing required config key in YAML: panel_end_date"):
                self._run_with_config(config_path)

    def test_config_requires_yaml_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = tmp_root / "pipeline_config.json"
            config_path.write_text('{"train_as_of_dates": "2011-11-09"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Engine config file must use .yaml or .yml"):
                self._run_with_config(config_path)

    def test_prediction_window_rejects_unsupported_main_pipeline_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = self._write_yaml(
                tmp_root,
                "pipeline_config.yaml",
                self._base_required_config().replace("prediction_window_days: 30\n", "prediction_window_days: 45\n"),
            )
            with self.assertRaisesRegex(ValueError, "--prediction-window-days must be one of"):
                self._run_with_config(config_path)

    def test_feature_lookback_window_rejects_unsupported_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = self._write_yaml(
                tmp_root,
                "pipeline_config.yaml",
                self._base_required_config().replace("feature_lookback_days: 90\n", "feature_lookback_days: 30\n"),
            )
            with self.assertRaisesRegex(ValueError, "--feature-lookback-days must be one of"):
                self._run_with_config(config_path)

    def test_feature_lookback_window_rejects_not_yet_wired_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            config_path = self._write_yaml(
                tmp_root,
                "pipeline_config.yaml",
                self._base_required_config().replace("feature_lookback_days: 90\n", "feature_lookback_days: 150\n"),
            )
            with self.assertRaisesRegex(ValueError, "--feature-lookback-days must be one of"):
                self._run_with_config(config_path)

    # ===== Config-Driven Orchestration =====
    def test_config_file_executes_pipeline_without_long_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            output_root = tmp_root / "data"
            gold_root = output_root / "gold" / "feature_store" / "purchase_propensity" / "propensity_train_dataset"
            config_path = tmp_root / "pipeline_config.yaml"
            config_path.write_text(
                self._base_required_config(output_root=output_root),
                encoding="utf-8",
            )
            for as_of_date in [snapshot.isoformat() for snapshot in generate_snapshot_dates(run_pipeline.date.fromisoformat(self.PANEL_END_DATE))]:
                dataset_path = gold_root / f"as_of_date={as_of_date}" / "propensity_train_dataset.parquet"
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                dataset_path.touch()

            with patch("sys.argv", ["run_pipeline.py", "--config", str(config_path)]):
                with patch("mle_marketplace_growth.purchase_propensity.run_pipeline.run_training") as mock_train:
                    with patch("mle_marketplace_growth.purchase_propensity.run_pipeline.run_policy_budget_evaluation") as mock_policy_eval:
                        with patch("mle_marketplace_growth.purchase_propensity.run_pipeline.run_validation", return_value=(True, {"checks": []})):
                            with patch("mle_marketplace_growth.purchase_propensity.run_pipeline.write_interpretation"):
                                run_pipeline.main()
            self.assertTrue(mock_train.called)
            self.assertEqual(mock_policy_eval.call_count, 2)


if __name__ == "__main__":
    unittest.main()
