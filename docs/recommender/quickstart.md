# Recommender Quickstart

Environment setup is shared at repository root `README.md`.

Datetime/global-vs-engine config strategy is documented in `docs/README.md`.

## Optional Tuning Sweep

Sweep scope (small fixed grid):

| Item | Behavior |
|---|---|
| Swept knobs | two-tower: `temperature`, `negative_samples`, `batch_size`, `early_stop_tolerance` |
| Non-swept knobs | inherited from `configs/recommender/default.yaml` |
| Baseline trial | `trial_default` runs first using current YAML values |
| Tuning summary | reports overall best trial by selected-model `Recall@20` and best two-tower trial by two-tower validation `Recall@20` |

```bash
# Shared layer (run once; reuse across engines)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_shared_silver --shared-config configs/shared.yaml

# Recommender gold layer (reads same engine config)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_gold_recommender --config configs/recommender/default.yaml

# Fixed small-grid tuning sweep (heavier than main run)
PYTHONPATH=src python scripts/tune_recommender_minimal.py --config configs/recommender/default.yaml
```

## Recommended Run

Pipeline execution is intentionally modular. Each pipeline stage can be run independently to simplify experimentation and reduce orchestration complexity for this demo repository.

### 1) Build Features

```bash
# Shared layer (run once; reuse across engines)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_shared_silver --shared-config configs/shared.yaml

# Recommender gold layer (reads same engine config)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_gold_recommender --config configs/recommender/default.yaml
```

### 2) Train Offline (with validation/test metrics + artifact checks)

```bash
# Train and offline-evaluate popularity/MF/two-tower from prebuilt gold
PYTHONPATH=src python -m mle_marketplace_growth.recommender.train \
  --splits-path data/gold/feature_store/recommender/user_item_splits/user_item_splits.parquet \
  --user-index-path data/gold/feature_store/recommender/user_index/user_index.parquet \
  --item-index-path data/gold/feature_store/recommender/item_index/item_index.parquet \
  --output-dir artifacts/recommender
```

Validation note:
- `validate_outputs` is core and runs automatically when using `run_pipeline`.
- direct `validate_outputs` CLI is only for manual debug/recovery runs.

### 3) Serve Batch (predict only; no split logic and no offline evaluation inside scoring)

```bash
# Generate Top-K candidates from frozen model bundle
PYTHONPATH=src python -m mle_marketplace_growth.recommender.predict \
  --model-bundle artifacts/recommender/model_bundle.pkl \
  --output-csv artifacts/recommender/topk_recommendations.csv \
  --top-k 20
```

### 4) Demo Wrapper (convenience path)

```bash
# End-to-end convenience wrapper (train + predict + validate)
PYTHONPATH=src python -m mle_marketplace_growth.recommender.run_pipeline --config configs/recommender/default.yaml

# Regenerate Recall@20 comparison chart used in analysis report
PYTHONPATH=src python scripts/report_recommender_recall_chart.py
```

Prescribed default path for this repo: section **4) Demo Wrapper**.

Fail-fast behavior:
- no optional paths in the default workflow (fixed time window + ANN-required serving).
- missing dependencies still fail immediately during module import/execution.
- engine-specific gold build requires prebuilt shared silver; run `build_shared_silver` first.
- ML pipeline consumes prebuilt recommender gold tables from `build_gold_recommender`.
- recommender event-date bounds are validated against available shared silver event-date bounds during feature-store build.
Contract details (split/model/artifact/acceptance): `docs/recommender/spec.md`.

## Outputs To Review

| Priority | Artifact(s) | Why |
|---|---|---|
| Must | `artifacts/recommender/output_validation_summary.json` | Confirms artifact contract and health checks |
| Must | `artifacts/recommender/output_interpretation.md` | Fast narrative summary of run outcomes |
| Must | `artifacts/recommender/validation_retrieval_metrics.json`, `artifacts/recommender/test_retrieval_metrics.json` | Core retrieval quality evidence |
| Must | `artifacts/recommender/topk_recommendations.csv` | Serving-style top-K output |
| Must (manual) | `docs/recommender/analysis_report.md` | Refresh report after rerun |

## Tests

Unit tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_recommender_minimal.py
```

Design/spec reference:
- `docs/recommender/spec.md`
