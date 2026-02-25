# Quickstart

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Recommended Steps

1) Run the manual two-cycle demo flow in **Recommended Runs** below.  
2) Review each cycle's `output_interpretation.md` and core policy artifacts in **Outputs To Review**.
3) Update `docs/purchase_propensity/analysis_report.md` with cycle comparison and final decisions (manual review or AI-assisted draft review from the artifacts).

## Recommended Runs

`run_pipeline` orchestrates feature-store build, train, predict, evaluate, and offline policy evaluation in one command.
It also runs automated artifact checks by default and writes:
- `<artifacts-dir>/output_validation_summary.json`
- `<artifacts-dir>/output_interpretation.md`
Recommended temporal schedule is documented in `docs/purchase_propensity/snapshot_plan.md`.

Two-cycle demo flow (manual, config-driven):

```bash
# Cycle 1: initial batch (first-year panel)
PYTHONPATH=src python -m mle_marketplace_growth.purchase_propensity.run_pipeline --config configs/purchase_propensity/demo_cycle_initial.yaml --artifacts-dir artifacts/purchase_propensity/cycle_initial

# Cycle 2: single rolling retrain
PYTHONPATH=src python -m mle_marketplace_growth.purchase_propensity.run_pipeline --config configs/purchase_propensity/demo_cycle_retrain.yaml --artifacts-dir artifacts/purchase_propensity/cycle_retrain
```

Two-cycle demo outputs:
- `artifacts/purchase_propensity/cycle_initial/...`
- `artifacts/purchase_propensity/cycle_retrain/...`
- Compare both cycles manually in `docs/purchase_propensity/analysis_report.md`.

Notes:
- `run_pipeline.py` uses monthly snapshot cadence for generated date panels.
- Config allows `prediction_window_days` `{30,60,90}` and `feature_lookback_days` `{60,90,120}`.
- Initial cycle uses `window_selection_mode=sensitivity` to freeze structural decisions; retrain cycle uses `window_selection_mode=fixed` to avoid reopening structural search.
- Model and policy design details are in `docs/purchase_propensity/spec.md`.

Optional remove existing outputs before clean rebuild:

```bash
rm -rf artifacts/purchase_propensity/*
rm -rf data/gold/feature_store/purchase_propensity/*
```

## Outputs To Review

- Must review (both cycles):
  - `output_validation_summary.json` (artifact sanity checks)
  - `output_interpretation.md` (auto narrative summary)
  - `offline_policy_budget_test.json` (final policy comparison used in report)
- Optional deep dive: `train_metrics.json`, `prediction_scores.csv`, `report_assets/policy_comparison_cycles.png`, and (initial cycle only) `window_sensitivity.json` + `window_validation_dashboard.png`.

## Optional Checks

Optional report chart regeneration (auditable/reproducible):

```bash
PYTHONPATH=src python scripts/report_policy_comparison_chart.py
```

Unit tests (fast checks):

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_run_pipeline.py tests/test_validate_outputs.py
```

Integration test (recommended flow on a small fixture):

```bash
.venv/bin/python -m unittest tests/test_purchase_propensity_integration.py
```

Integration test output behavior:
- `tests/test_purchase_propensity_integration.py` writes to a temporary `/tmp/...` folder.
- It does not overwrite your main `data/` or `artifacts/purchase_propensity/` outputs.
