"""Validate purchase propensity artifacts with automated sanity checks."""

import argparse
import json
from pathlib import Path


# ===== Artifact Loaders =====
def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ===== Validation Checks =====
def run_validation(
    artifacts_dir: Path,
    expect_window_sensitivity: bool,
    output_json: Path | None = None,
) -> tuple[bool, dict]:
    # Load required artifacts.
    train_metrics = _load_json(artifacts_dir / "train_metrics.json")
    budget_eval_test_path = artifacts_dir / "offline_policy_budget_test.json"
    budget_eval_validation_path = artifacts_dir / "offline_policy_budget_validation.json"
    budget_eval_test = _load_json(budget_eval_test_path)

    checks: list[dict] = []

    # Model and metric consistency checks.
    selected_model_name = train_metrics.get("selected_model_name")
    checks.append(
        {
            "check": "selected_model_valid",
            "description": "Selected model must be one of the supported candidates.",
            "passed": selected_model_name in {"logistic_regression", "xgboost"},
            "detail": f"selected_model_name={selected_model_name}",
        }
    )

    candidate_metrics = train_metrics.get("propensity_model_candidates", [])
    candidate_bounds_ok = all(
        0.0 <= float(row.get("roc_auc", -1.0)) <= 1.0 and 0.0 <= float(row.get("average_precision", -1.0)) <= 1.0
        for row in candidate_metrics
    )
    checks.append(
        {
            "check": "candidate_metrics_in_bounds",
            "description": "Model metrics should be valid probabilities in [0, 1].",
            "passed": candidate_bounds_ok,
            "detail": f"candidate_count={len(candidate_metrics)}",
        }
    )

    by_policy = {row.get("policy"): row for row in budget_eval_test.get("policy_comparison", [])}
    ml_revenue = float(by_policy.get("ml_top_expected_value", {}).get("actual_revenue_per_targeted_user", 0.0))
    random_revenue = float(by_policy.get("random_baseline", {}).get("actual_revenue_per_targeted_user", 0.0))
    checks.append(
        {
            "check": "ml_beats_random_revenue_per_user",
            "description": "ML targeting should match or outperform random baseline on holdout revenue per targeted user.",
            "passed": ml_revenue >= random_revenue,
            "detail": f"ml={ml_revenue:.6f}, random={random_revenue:.6f}",
        }
    )
    budget_eval_validation = _load_json(budget_eval_validation_path)
    checks.append(
        {
            "check": "budget_policy_validation_and_test_outputs_present",
            "description": "Budget-constrained policy evaluation should include ML/Random/RFM for validation and test slices.",
            "passed": len(budget_eval_validation.get("policy_comparison", [])) == 3
            and len(budget_eval_test.get("policy_comparison", [])) == 3,
            "detail": (
                f"validation_policies={len(budget_eval_validation.get('policy_comparison', []))}, "
                f"test_policies={len(budget_eval_test.get('policy_comparison', []))}"
            ),
        }
    )

    ml_policy = by_policy.get("ml_top_expected_value", {})
    checks.append(
        {
            "check": "budget_policy_selection_nonzero",
            "description": "Budget-constrained policy comparison must target at least one user.",
            "passed": int(ml_policy.get("targeted_users", 0)) > 0 and float(ml_policy.get("budget_spend", 0.0)) > 0.0,
            "detail": f"targeted_users={ml_policy.get('targeted_users')}, budget_spend={ml_policy.get('budget_spend')}",
        }
    )

    # Optional sensitivity artifact checks.
    sensitivity_path = artifacts_dir / "window_sensitivity.json"
    if expect_window_sensitivity:
        sensitivity = _load_json(sensitivity_path)
        windows = [int(row.get("window_days", -1)) for row in sensitivity.get("window_sensitivity", [])]
        checks.append(
            {
                "check": "window_sensitivity_has_30_60_90",
                "description": "Sensitivity output should include windows 30, 60, and 90 days.",
                "passed": windows == [30, 60, 90],
                "detail": f"windows={windows}",
            }
        )

    passed = all(row["passed"] for row in checks)
    summary = {"passed": passed, "artifacts_dir": str(artifacts_dir), "checks": checks}

    # Optional summary output.
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return passed, summary


# ===== Automated Interpretation =====
def write_interpretation(
    artifacts_dir: Path,
    output_md: Path | None = None,
) -> Path:
    train_metrics = _load_json(artifacts_dir / "train_metrics.json")
    budget_eval_test = _load_json(artifacts_dir / "offline_policy_budget_test.json")
    budget_eval_validation = _load_json(artifacts_dir / "offline_policy_budget_validation.json")

    validation_quality = train_metrics.get("validation_quality", {})
    top_lift = float(validation_quality.get("top_decile_lift", 0.0))
    pr_auc = float(validation_quality.get("average_precision", 0.0))
    roc_auc = float(validation_quality.get("roc_auc", 0.0))
    ece = float(validation_quality.get("ece_10_bin", 0.0))
    brier = float(validation_quality.get("brier_score", 1.0))

    policies = {row["policy"]: row for row in budget_eval_test.get("policy_comparison", [])}
    policies_validation = {row["policy"]: row for row in budget_eval_validation.get("policy_comparison", [])}
    ml_rev = float(policies.get("ml_top_expected_value", {}).get("actual_revenue_per_targeted_user", 0.0))
    random_rev = float(policies.get("random_baseline", {}).get("actual_revenue_per_targeted_user", 0.0))
    rfm_rev = float(policies.get("rfm_heuristic", {}).get("actual_revenue_per_targeted_user", 0.0))

    window_summary = "Window sensitivity not available."
    sensitivity_path = artifacts_dir / "window_sensitivity.json"
    if sensitivity_path.exists():
        sensitivity = _load_json(sensitivity_path)
        window_rows = sensitivity.get("window_sensitivity", [])
        if window_rows:
            best_window = max(
                window_rows,
                key=lambda row: max(item["average_precision"] for item in row.get("model_results", [{"average_precision": 0.0}])),
            )
            best_window_ap = max(item["average_precision"] for item in best_window.get("model_results", []))
            window_summary = (
                f"Best prediction window by PR-AUC: {best_window['window_days']}d "
                f"(PR-AUC={best_window_ap:.4f})."
            )

    lines = [
        "# Automated Interpretation",
        "",
        "## Model Generalization",
        f"- ROC-AUC: {roc_auc:.4f}",
        f"- PR-AUC: {pr_auc:.4f}",
        f"- Top-decile lift: {top_lift:.4f}",
        f"- Calibration (ECE, lower is better): {ece:.4f}",
        f"- Calibration (Brier, lower is better): {brier:.4f}",
        "",
        "## Policy Comparison (Budget-Constrained Holdout Outcomes)",
        f"- ML expected-value revenue/targeted user: {ml_rev:.4f}",
        f"- Random baseline revenue/targeted user: {random_rev:.4f}",
        f"- RFM baseline revenue/targeted user: {rfm_rev:.4f}",
        f"- ML vs Random delta: {(ml_rev - random_rev):.4f}",
        f"- ML vs RFM delta: {(ml_rev - rfm_rev):.4f}",
        f"- Validation policy rows: {len(budget_eval_validation.get('policy_comparison', []))}",
        f"- Test policy rows: {len(budget_eval_test.get('policy_comparison', []))}",
        "",
        "## Budget Policy Summary",
        "- Policy comparison uses equal budget-constrained Top-K across ML expected value, random baseline, and RFM baseline.",
        "- This section reports offline holdout outcomes only; no causal incrementality claim.",
        f"- ML targeted users (validation): {int(policies_validation.get('ml_top_expected_value', {}).get('targeted_users', 0))}",
        f"- ML targeted users (test): {int(policies.get('ml_top_expected_value', {}).get('targeted_users', 0))}",
        "",
        "## Window Validation",
        f"- {window_summary}",
        "- Use this for model-signal comparison only; it is not an automatic business-window selection rule.",
        "",
        "_Scope note: offline predictive policy evaluation only; not causal promotional incrementality._",
        "",
    ]

    report_path = output_md or (artifacts_dir / "output_interpretation.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ===== Entry Point =====
def main() -> None:
    # ===== CLI Arguments =====
    parser = argparse.ArgumentParser(description="Validate generated purchase propensity artifacts.")
    parser.add_argument("--artifacts-dir", default="artifacts/purchase_propensity", help="Directory containing generated artifacts")
    parser.add_argument("--output-json", default="artifacts/purchase_propensity/output_validation_summary.json", help="Where to write output-validation summary JSON")
    parser.add_argument("--expect-window-sensitivity", action="store_true", help="Require window_sensitivity.json with 30/60/90 windows")
    args = parser.parse_args()

    # ===== Validate + Write Interpretation =====
    passed, summary = run_validation(artifacts_dir=Path(args.artifacts_dir), expect_window_sensitivity=args.expect_window_sensitivity, output_json=Path(args.output_json))
    if not passed:
        failed = [row for row in summary["checks"] if not row["passed"]]
        raise SystemExit(f"Validation failed: {failed}")
    interpretation_path = write_interpretation(artifacts_dir=Path(args.artifacts_dir))
    print(f"Wrote interpretation: {interpretation_path}")
    print(f"Validation passed: {args.output_json}")


if __name__ == "__main__":
    main()
