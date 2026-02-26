# Purchase Propensity Quickstart

Environment setup is shared at repository root `README.md`.
Datetime/global-vs-engine config strategy is documented in `docs/README.md`.

## Recommended Steps

1) Run the manual two-cycle demo flow in **Recommended Runs** below.
2) Review each cycle's `output_interpretation.md` and core policy artifacts in **Outputs To Review**.
3) Update `docs/purchase_propensity/analysis_report.md` with cycle comparison and final decisions (manual review or AI-assisted draft review from artifacts).

## Recommended Runs

`run_pipeline` orchestrates feature-store build, train, predict, evaluate, and offline policy evaluation.
It writes:
- `<artifacts-dir>/output_validation_summary.json`
- `<artifacts-dir>/output_interpretation.md`

Two-cycle demo flow (manual, config-driven):

```bash
# Shared layer (run once; reuse across engines/cycles)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build --shared-config configs/shared.yaml --build-engines shared

# Cycle 1: initial batch (first-year panel)
PYTHONPATH=src python -m mle_marketplace_growth.purchase_propensity.run_pipeline --config configs/purchase_propensity/demo_cycle_initial.yaml --artifacts-dir artifacts/purchase_propensity/cycle_initial

# Cycle 2: single rolling retrain
PYTHONPATH=src python -m mle_marketplace_growth.purchase_propensity.run_pipeline --config configs/purchase_propensity/demo_cycle_retrain.yaml --artifacts-dir artifacts/purchase_propensity/cycle_retrain

# Regenerate policy comparison chart used in analysis report
PYTHONPATH=src python scripts/report_policy_comparison_chart.py
```

Notes:
- Initial cycle uses `window_selection_mode=sensitivity` to freeze structural decisions.
- Retrain cycle uses `window_selection_mode=fixed` to avoid reopening structural search.
- engine-specific gold build requires prebuilt shared silver; run `--build-engines shared` first.
- cycle dates are validated against available shared silver event-date bounds during feature-store build.
- Design details: `docs/purchase_propensity/spec.md`.

Optional clean rebuild:

```bash
rm -rf artifacts/purchase_propensity/*
rm -rf data/gold/feature_store/purchase_propensity/*
```

## Outputs To Review

- Must review (both cycles):
  - `output_validation_summary.json`
  - `output_interpretation.md`
  - `offline_policy_budget_test.json`
- Optional deep dive:
  - `train_metrics.json`
  - `prediction_scores.csv`
  - `report_assets/policy_comparison_cycles.png`
  - initial cycle only: `window_sensitivity.json`, `window_validation_dashboard.png`

## Optional Checks

Unit tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_run_pipeline.py tests/test_validate_outputs.py tests/test_purchase_propensity_minimal.py
```

Integration test:

```bash
.venv/bin/python -m unittest tests/test_purchase_propensity_integration.py
```
