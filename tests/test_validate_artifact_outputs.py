import json
import tempfile
import unittest
from pathlib import Path

from mle_marketplace_growth.purchase_propensity.validate_artifact_outputs import run_validation


class ValidateOutputsTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        # Compact helper for per-test artifact payloads.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_run_validation_pass(self) -> None:
        # Happy path: supported model + full policy rows + sensitivity coverage.
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = Path(tmp_dir)
            self._write_json(
                artifacts / "train_metrics.json",
                {
                    "selected_model_name": "logistic_regression",
                    "propensity_model_candidates": [
                        {"model_name": "logistic_regression", "roc_auc": 0.7, "average_precision": 0.5}
                    ],
                },
            )
            self._write_json(
                artifacts / "offline_policy_budget_validation.json",
                {
                    "policy_comparison": [
                        {"policy": "ml_top_expected_value", "actual_revenue_per_targeted_user": 10.0, "targeted_users": 100, "budget_spend": 500.0},
                        {"policy": "random_baseline", "actual_revenue_per_targeted_user": 5.0, "targeted_users": 100, "budget_spend": 500.0},
                        {"policy": "rfm_heuristic", "actual_revenue_per_targeted_user": 9.0, "targeted_users": 100, "budget_spend": 500.0},
                    ]
                },
            )
            self._write_json(
                artifacts / "offline_policy_budget_test.json",
                {
                    "policy_comparison": [
                        {"policy": "ml_top_expected_value", "actual_revenue_per_targeted_user": 10.0, "targeted_users": 100, "budget_spend": 500.0},
                        {"policy": "random_baseline", "actual_revenue_per_targeted_user": 5.0, "targeted_users": 100, "budget_spend": 500.0},
                        {"policy": "rfm_heuristic", "actual_revenue_per_targeted_user": 9.0, "targeted_users": 100, "budget_spend": 500.0},
                    ]
                },
            )
            self._write_json(
                artifacts / "window_sensitivity.json",
                {"window_sensitivity": [{"window_days": 30}, {"window_days": 60}, {"window_days": 90}]},
            )

            passed, summary = run_validation(
                artifacts_dir=artifacts,
                expect_window_sensitivity=True,
                output_json=artifacts / "output_validation_summary.json",
            )
            self.assertTrue(passed)
            self.assertTrue((artifacts / "output_validation_summary.json").exists())
            self.assertTrue(all(row["passed"] for row in summary["checks"]))

    def test_run_validation_fail(self) -> None:
        # Failure path: invalid model/metrics and incomplete policy outputs.
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = Path(tmp_dir)
            self._write_json(
                artifacts / "train_metrics.json",
                {
                    "selected_model_name": "bad_model",
                    "propensity_model_candidates": [{"model_name": "bad_model", "roc_auc": 1.2, "average_precision": -0.1}],
                },
            )
            self._write_json(
                artifacts / "offline_policy_budget_validation.json",
                {
                    "policy_comparison": [
                        {"policy": "ml_top_expected_value", "actual_revenue_per_targeted_user": 4.0, "targeted_users": 0, "budget_spend": 0.0},
                        {"policy": "random_baseline", "actual_revenue_per_targeted_user": 5.0, "targeted_users": 0, "budget_spend": 0.0},
                    ]
                },
            )
            self._write_json(
                artifacts / "offline_policy_budget_test.json",
                {
                    "policy_comparison": [
                        {"policy": "ml_top_expected_value", "actual_revenue_per_targeted_user": 4.0, "targeted_users": 0, "budget_spend": 0.0},
                        {"policy": "random_baseline", "actual_revenue_per_targeted_user": 5.0, "targeted_users": 0, "budget_spend": 0.0},
                    ]
                },
            )
            self._write_json(
                artifacts / "window_sensitivity.json",
                {"window_sensitivity": [{"window_days": 30}, {"window_days": 60}, {"window_days": 90}]},
            )

            passed, summary = run_validation(artifacts_dir=artifacts, expect_window_sensitivity=True)
            self.assertFalse(passed)
            self.assertTrue(any(not row["passed"] for row in summary["checks"]))


if __name__ == "__main__":
    unittest.main()
