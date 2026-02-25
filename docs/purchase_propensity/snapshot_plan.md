# Purchase Propensity Snapshot Plan

Purpose: define a reproducible time-based evaluation schedule for out-of-time validation.

## Principles

- Use multiple `as_of_date` snapshots (not a single split) to reduce overfitting to one period.
- Keep snapshots strictly time-ordered.
- Reserve the latest snapshot(s) for final test-style interpretation.
- Use explicit cycle configs for reproducibility.
- Keep main pipeline windows unchanged (`90d` lookback features, `30d` purchase label) across snapshots.
- For production operations, apply the same pattern on a rolling quarterly retraining cadence.

## Default Plan (Monthly)

- **Training/validation panel snapshots:** monthly `as_of_date` from `2010-06-09` to `2011-11-09`.
- **Scoring/evaluation snapshot:** `2011-11-09`.
- **Sensitivity snapshot:** `2011-09-09` (to avoid right-censoring for 60/90-day windows).
- Strict architecture-aligned cycle option: use exactly 12 monthly snapshots; `run_pipeline.py` enforces a fixed 10 train / 1 validation / 1 test split.

Why these dates:
- Early enough to include broad historical variation.
- Late enough to preserve enough future rows for label windows.
- Matches current main-pipeline guardrails (`30d` target, `90d` lookback).

## Example Command Pattern

Use `run_pipeline` with a cycle config:

```bash
PYTHONPATH=src python -m mle_marketplace_growth.purchase_propensity.run_pipeline --config configs/purchase_propensity/demo_cycle_initial.yaml --artifacts-dir artifacts/purchase_propensity/cycle_initial
```

## Review Expectation

- Use each cycle's `output_validation_summary.json` as automated gate.
- Then review each cycle's budget policy outputs and interpretation markdown for business comparison.
