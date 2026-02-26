import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


def _write_fixture_raw_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]
    rows = []
    invoice_id = 1
    for user_idx in range(1, 81):
        user_id = f"{10000 + user_idx}.0"
        country = "United Kingdom" if user_idx % 2 == 0 else "France"
        historical_events = [
            ("2010-12-10 10:00:00", 1, 9.0),
            ("2011-01-10 10:00:00", 1, 9.5),
            ("2011-02-10 10:00:00", 1, 10.0),
            ("2011-03-10 10:00:00", 1, 10.5),
            ("2011-04-10 10:00:00", 1, 11.0),
            ("2011-05-10 10:00:00", 1, 11.5),
            ("2011-06-10 10:00:00", 1, 12.0),
            ("2011-07-10 10:00:00", 2, 12.5),
            ("2011-08-20 10:00:00", 1, 15.0),
            ("2011-09-10 10:00:00", 1, 13.0),
            ("2011-10-10 10:00:00", 1, 13.5),
        ]
        if user_idx % 2 == 0:
            historical_events.append(("2011-11-10 10:00:00", 1, 14.0))
        if user_idx % 2 == 0:
            historical_events.append(("2011-10-20 10:00:00", 1, 18.0))
        if user_idx <= 3:
            historical_events.append(("2011-10-25 10:00:00", 1, 5000.0))
        for event_ts, quantity, price in historical_events:
            rows.append(
                [
                    str(invoice_id),
                    f"SKU{user_idx:04d}",
                    "Gift Item",
                    str(quantity),
                    event_ts,
                    str(price),
                    user_id,
                    country,
                ]
            )
            invoice_id += 1

        if user_idx % 4 in {0, 1}:
            rows.append(
                [
                    str(invoice_id),
                    f"SKU{user_idx:04d}",
                    "Gift Item",
                    "1",
                    "2011-09-25 10:00:00",
                    "20.0",
                    user_id,
                    country,
                ]
            )
            invoice_id += 1
        if user_idx % 5 == 0:
            rows.append(
                [
                    str(invoice_id),
                    f"SKU{user_idx:04d}",
                    "Gift Item",
                    "1",
                    "2011-10-25 10:00:00",
                    "25.0",
                    user_id,
                    country,
                ]
            )
            invoice_id += 1
        if user_idx % 6 == 0:
            rows.append(
                [
                    str(invoice_id),
                    f"SKU{user_idx:04d}",
                    "Gift Item",
                    "1",
                    "2011-11-20 10:00:00",
                    "30.0",
                    user_id,
                    country,
                ]
            )
            invoice_id += 1
        if user_idx % 3 == 0:
            rows.append(
                [
                    str(invoice_id),
                    f"SKU{user_idx:04d}",
                    "Gift Item",
                    "1",
                    "2011-11-25 10:00:00",
                    "35.0",
                    user_id,
                    country,
                ]
            )
            invoice_id += 1

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


class PurchasePropensityPipelineIntegrationTest(unittest.TestCase):
    def _run(self, command: list[str], env: dict[str, str]) -> None:
        subprocess.run(command, check=True, env=env, cwd=Path(__file__).resolve().parents[1])

    def test_end_to_end_recommended_flow(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_csv = tmp_root / "raw.csv"
            output_root = tmp_root / "data"
            artifacts_root = tmp_root / "artifacts"
            config_path = tmp_root / "config.yaml"
            _write_fixture_raw_csv(input_csv)

            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root / "src")
            env["OMP_NUM_THREADS"] = "1"

            config_path.write_text(
                "\n".join(
                    [
                        f"input_csv: \"{input_csv}\"",
                        f"output_root: \"{output_root}\"",
                        f"artifacts_dir: \"{artifacts_root}\"",
                        "train_start_date: \"2010-12-10\"",
                        "train_end_date: \"2011-11-10\"",
                        "score_as_of_date: \"2011-11-10\"",
                        "prediction_window_days: 30",
                        "feature_lookback_days: 90",
                        "budget: 500",
                        "cost_per_user: 5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self._run(
                [
                    ".venv/bin/python",
                    "-m",
                    "mle_marketplace_growth.feature_store.build",
                    "--build-engines",
                    "shared",
                    "--input-csv",
                    str(input_csv),
                    "--output-root",
                    str(output_root),
                ],
                env,
            )
            self._run(
                [
                    ".venv/bin/python",
                    "-m",
                    "mle_marketplace_growth.purchase_propensity.run_pipeline",
                    "--config",
                    str(config_path),
                ],
                env,
            )

            train_metrics = json.loads((artifacts_root / "train_metrics.json").read_text(encoding="utf-8"))
            self.assertIn(train_metrics["selected_model_name"], {"logistic_regression", "xgboost"})
            self.assertEqual(train_metrics["calibration_method"], "sigmoid")
            self.assertGreater(train_metrics["spend_cap_value"], 0)
            self.assertGreater(train_metrics["validation_quality"]["top_decile_lift"], 0)
            self.assertGreater(train_metrics["test_quality"]["top_decile_lift"], 0)
            self.assertGreater(train_metrics["test_rows"], 0)
            budget_eval_validation = json.loads((artifacts_root / "offline_policy_budget_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(len(budget_eval_validation["policy_comparison"]), 3)
            budget_eval_test = json.loads((artifacts_root / "offline_policy_budget_test.json").read_text(encoding="utf-8"))
            self.assertEqual(len(budget_eval_test["policy_comparison"]), 3)

            sensitivity = json.loads((artifacts_root / "window_sensitivity.json").read_text(encoding="utf-8"))
            self.assertEqual([row["window_days"] for row in sensitivity["window_sensitivity"]], [30, 60, 90])
            self.assertTrue(sensitivity["prediction_window_validation"])
            self.assertEqual(
                [row["feature_lookback_days"] for row in sensitivity["feature_window_validation"]],
                [60, 90, 120],
            )


if __name__ == "__main__":
    unittest.main()
