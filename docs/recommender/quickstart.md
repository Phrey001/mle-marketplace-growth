# Recommender Quickstart

Environment setup is shared at repository root `README.md`.

Datetime/global-vs-engine config strategy is documented in `docs/README.md`.
Feature store data debugging tips (including parquet → CSV export commands) live in `docs/feature_store/overview.md`.

## Recommended Steps

1) Run the deterministic recommender demo flow in **Recommended Run** below.
2) Review `output_interpretation.md` and the core retrieval artifacts in **Outputs To Review**.
3) Refresh the report chart and update `docs/recommender/analysis_report.md` after reruns with material result changes.

## Recommended Run

Use the deterministic end-to-end wrapper as the prescribed default path for this demo repo.

### 1) Build Features

```bash
# Shared layer (run once; reuse across engines)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_shared_silver --shared-config configs/shared.yaml

# Recommender gold layer (reads same engine config)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_gold_recommender --config configs/recommender/default.yaml
```

Feature-store note:
- Run `build_shared_silver` before `build_gold_recommender`.

### 2) Run Pipeline

```bash
# End-to-end recommender ML pipeline (train + predict + validate)
PYTHONPATH=src python -m mle_marketplace_growth.recommender.run_pipeline --config configs/recommender/default.yaml
```

Pipeline note:
- `run_pipeline.py` consumes the prebuilt recommender gold tables from `build_gold_recommender`.
- Recommender feature-store outputs keep one canonical latest build; reruns overwrite prior outputs.

### 3) Refresh Report Chart

```bash
# Regenerate Recall@20 comparison chart used in the analysis report
PYTHONPATH=src python scripts/report_recommender_recall_chart.py --config configs/recommender/default.yaml
```

### 4) Update Analysis Report

Update [analysis_report.md](/home/phrey/projects/mle-marketplace-growth/docs/recommender/analysis_report.md) after reruns with material result changes (manual review or AI-assisted draft review from artifacts).

Prescribed default path for this repo: steps **1) Build Features**, **2) Run Pipeline**, **3) Refresh Report Chart**, and **4) Update Analysis Report**.

## Optional ML Pipeline Steps

Use the commands below only if you want to inspect or rerun the offline ML pipeline stages one at a time instead of the prescribed `run_pipeline.py` path.

### 1) Train

```bash
# Train and offline-evaluate popularity/MF/two-tower from prebuilt gold
PYTHONPATH=src python -m mle_marketplace_growth.recommender.train --config configs/recommender/default.yaml
```

### 2) Predict

```bash
# Generate serving retrieval artifacts + Top-K candidates from frozen model bundle
PYTHONPATH=src python -m mle_marketplace_growth.recommender.predict --config configs/recommender/default.yaml
```

### 3) Validate Outputs

```bash
# Validate artifact contract and write interpretation summary
PYTHONPATH=src python -m mle_marketplace_growth.recommender.validate_outputs --config configs/recommender/default.yaml
```

## Optional Serving Path

The standalone serving command is the same `Predict` step shown above. This section exists only as a quick reference because serving is a common review question.

```bash
# Generate serving retrieval artifacts + Top-K candidates from frozen model bundle
PYTHONPATH=src python -m mle_marketplace_growth.recommender.predict --config configs/recommender/default.yaml
```

Serving notes:
- This is the batch-serving-style step after training has already frozen the selected model in `model_bundle.pkl`.
- The script derives the canonical artifact folder from `recommender_max_event_date` in the same YAML config, then reads `artifacts/recommender/as_of=<recommender_max_event_date>/model_bundle.pkl`.
- It reads the saved model artifacts, builds ANN retrieval artifacts, and writes `topk_recommendations.csv`.
- `run_pipeline.py` already includes this step so the recommended demo flow produces the final serving-style recommendation output in one run.
- It is suitable as a standalone batch-scoring script for this demo repo, but not a full production deployment system by itself.

## Outputs To Review

All artifacts are written under:
- `artifacts/recommender/as_of=<recommender_max_event_date>/`

| Priority | Artifact(s) | Why |
|---|---|---|
| Must | `artifacts/recommender/as_of=<recommender_max_event_date>/output_validation_summary.json` | Confirms artifact contract and health checks |
| Must | `artifacts/recommender/as_of=<recommender_max_event_date>/output_interpretation.md` | Fast narrative summary of run outcomes |
| Must | `artifacts/recommender/as_of=<recommender_max_event_date>/validation_retrieval_metrics.json`, `artifacts/recommender/as_of=<recommender_max_event_date>/test_retrieval_metrics.json` | Core retrieval quality evidence |
| Must | `artifacts/recommender/as_of=<recommender_max_event_date>/topk_recommendations.csv` | Serving-style top-K output |
| Must (manual) | `docs/recommender/analysis_report.md` | Refresh report after rerun |

## Optional Experiments

### Two-Tower Challenger Tuning Sweep

This sweep only tunes two-tower hyperparameters around the current default YAML settings.
It is optional low-priority experimentation and is not part of the main reporting flow.

| Item | Behavior |
|---|---|
| Sweep scope | two-tower challenger tuning only |
| Trial generation | `trial_default` uses the current YAML values exactly; each `trial_i` applies one fixed two-tower override set on top of those YAML defaults |
| Swept knobs | two-tower: `temperature`, `negative_samples`, `batch_size`, `early_stop_tolerance` |
| Non-swept knobs | popularity and MF settings stay inherited from `configs/recommender/default.yaml` |
| Baseline trial | `trial_default` runs first using the current YAML values as the two-tower baseline |
| Tuning summary | reports overall best trial by selected-model `Recall@20` and best two-tower trial by two-tower validation `Recall@20` |
| Output root | `artifacts/recommender/tuning/` |

```bash
# Shared layer (run once; reuse across engines)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_shared_silver --shared-config configs/shared.yaml

# Recommender gold layer (reads same engine config)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_gold_recommender --config configs/recommender/default.yaml

# Fixed small-grid two-tower tuning sweep (heavier than main run)
PYTHONPATH=src python scripts/tune_recommender_minimal.py --config configs/recommender/default.yaml
```

Review these tuning artifacts:
- `artifacts/recommender/tuning/tuning_summary.json`
- `artifacts/recommender/tuning/trial_default/`
- `artifacts/recommender/tuning/trial_*/`

Each trial folder contains:
- `trial_config.yaml`
- `validation_retrieval_metrics.json`
- `test_retrieval_metrics.json`

## Tests

Unit tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_recommender_minimal.py
```

Design/spec reference:
- `docs/recommender/spec.md`
