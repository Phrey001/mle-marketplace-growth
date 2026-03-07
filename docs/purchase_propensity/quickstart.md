# Purchase Propensity Quickstart

Environment setup is shared at repository root `README.md`.
Datetime/global-vs-engine config strategy is documented in `docs/README.md`.

## Recommended Steps

1) Run the manual two-cycle demo flow in **Recommended Runs** below.
2) Review each cycle's `output_interpretation.md` and core policy artifacts in **Outputs To Review**.
3) Update `docs/purchase_propensity/analysis_report.md` with cycle comparison and final decisions (manual review or AI-assisted draft review from artifacts).

## Recommended Runs

Pipeline execution is intentionally modular. Each pipeline stage can be run independently to simplify experimentation and reduce orchestration complexity for this demo repository.

### 1) Build Features

```bash
# Shared layer (run once; reuse across engines/cycles)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_shared_silver --shared-config configs/shared.yaml

# Cycle 1 gold layer (12 monthly snapshots from cycle config)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_gold_purchase_propensity --config configs/purchase_propensity/cycle_initial.yaml

# Cycle 2 gold layer (rolling 12-month overlap from cycle config)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_gold_purchase_propensity --config configs/purchase_propensity/cycle_retrain.yaml
```

### 2) Offline Pipeline (sensitivity -> train + model validation -> policy eval [validation + test] -> artifact quality checks)

Current implementation note:
- Purchase propensity offline train/eval uses strict 12-snapshot panel assembly and policy-budget evaluation.
- That orchestration is packaged in `run_pipeline` for this engine (this is why the command may look broader than a thin train-only wrapper).
- Inside `run_pipeline`, policy evaluation runs before final artifact validation checks.
- `validate_outputs` is core in this flow and runs automatically inside `run_pipeline` (standalone validator invocation is only for manual debug/recovery).

```bash
# Cycle 1 offline train/eval/validate convenience wrapper
PYTHONPATH=src python -m mle_marketplace_growth.purchase_propensity.run_pipeline --config configs/purchase_propensity/cycle_initial.yaml

# Cycle 2 offline train/eval/validate convenience wrapper
PYTHONPATH=src python -m mle_marketplace_growth.purchase_propensity.run_pipeline --config configs/purchase_propensity/cycle_retrain.yaml
```

Contract details (split/model/policy/artifact/acceptance): `docs/purchase_propensity/spec.md`.

Optional (explicit structural search run used by cycle 1 sensitivity mode):

```bash
# Run window sensitivity directly (standalone)
PYTHONPATH=src python -m mle_marketplace_growth.purchase_propensity.window_sensitivity \
  --panel-root data/gold/feature_store/purchase_propensity/propensity_train_dataset \
  --panel-end-date 2010-11-01 \
  --events-path data/silver/transactions_line_items/transactions_line_items.parquet \
  --output-json artifacts/purchase_propensity/cycle_initial/offline_eval/window_sensitivity.json \
  --output-plot artifacts/purchase_propensity/cycle_initial/offline_eval/window_validation_dashboard.png
```

### 2b) Refresh Report Chart (after offline train/evaluation artifacts are generated)

```bash
# Regenerate policy comparison chart used in analysis report
PYTHONPATH=src python scripts/report_policy_comparison_chart.py
```

### 3) Optional: Production Targeting Serve Batch (out of offline-eval scope)

```bash
# Score latest cycle 1 snapshot with frozen model artifact
PYTHONPATH=src python -m mle_marketplace_growth.purchase_propensity.predict \
  --input-path data/gold/feature_store/purchase_propensity/user_features_asof/as_of_date=2010-11-01/user_features_asof.parquet \
  --model-path artifacts/purchase_propensity/cycle_initial/offline_eval/propensity_model.pkl \
  --output-csv artifacts/purchase_propensity/serving_batch/as_of_date=2010-11-01/prediction_scores.csv

# Score latest cycle 2 snapshot with frozen model artifact
PYTHONPATH=src python -m mle_marketplace_growth.purchase_propensity.predict \
  --input-path data/gold/feature_store/purchase_propensity/user_features_asof/as_of_date=2011-02-01/user_features_asof.parquet \
  --model-path artifacts/purchase_propensity/cycle_retrain/offline_eval/propensity_model.pkl \
  --output-csv artifacts/purchase_propensity/serving_batch/as_of_date=2011-02-01/prediction_scores.csv
```

Stage 3 notes:
- Align `as_of_date` with each cycle config `panel_end_date`.
- Write outputs under `serving_batch/` to keep serving artifacts separate from offline evaluation artifacts.
- Purpose: produce production targeting scores when realized outcomes are not yet available.
- This stage is optional for this project because offline policy evaluation is fully covered by stages 1-2.

Notes:
| Item | Meaning |
|---|---|
| Cycle 1 mode | `window_selection_mode=sensitivity` to freeze structural decisions |
| Cycle 2 mode | `window_selection_mode=fixed` to avoid reopening structural search |
| Shared dependency | Engine-specific gold build requires prebuilt shared silver (`build_shared_silver`) |
| Gold dependency | ML pipeline consumes prebuilt purchase-propensity gold snapshots from `build_gold_purchase_propensity` |
| Artifact folder default | `--config cycle_initial.yaml` maps to `artifacts/purchase_propensity/cycle_initial` (same for retrain); override with `--artifacts-dir` only when needed |
| Date validation | Cycle dates are validated against shared silver event-date bounds |
| Artifact layout | Stage-2 outputs are grouped by purpose: `offline_eval/` (train/policy artifacts) and `report/` (validation summary + interpretation) |

Optional clean rebuild:

```bash
rm -rf artifacts/purchase_propensity/*
rm -rf data/gold/feature_store/purchase_propensity/*
```

## Outputs To Review

| Priority | Artifact(s) | Why |
|---|---|---|
| Recommended | `report/output_validation_summary.json` | Confirms run artifacts are internally valid |
| Recommended | `report/output_interpretation.md` | Fast narrative summary of run outcomes |
| Recommended | `offline_eval/offline_policy_budget_test.json` | Final test-slice policy comparison |
| Optional | `offline_eval/train_metrics.json` | Detailed model-quality diagnostics |
| Optional (stage 3 only) | `serving_batch/as_of_date=YYYY-MM-DD/prediction_scores.csv` | Serving-style scored user list (not required for core offline demo evaluation) |
| Optional | `report_assets/policy_comparison_cycles.png` | Report chart asset |
| Optional (cycle 1 only) | `offline_eval/window_sensitivity.json`, `offline_eval/window_validation_dashboard.png` | Structural search and freeze evidence |

## Optional Checks

Unit tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_run_pipeline.py tests/test_validate_outputs.py tests/test_purchase_propensity_minimal.py
```

Integration test:

```bash
.venv/bin/python -m unittest tests/test_purchase_propensity_integration.py
```
