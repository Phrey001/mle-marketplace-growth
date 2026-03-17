# Recommender — Analytical Report

## Report Metadata

- **Artifact source:** `artifacts/recommender/as_of=<recommender_max_event_date>/`
- **Run scope:** single-cycle recommender demo from the recommended quickstart flow
- **Update policy:** update this report (or append a new version) after reruns with material result changes

## 1) Executive Summary

- **Operational call to action:** keep `mf` as the current retrieval default, retain `popularity` as sanity baseline, and continue tuning `two_tower` as the challenger model.
- **Two-tower Challenger Path (out of current scope):** prioritize richer user/item features in the feature store (for example RFM-style user behavior signals and item metadata features), as the most practical next step to close the gap vs MF.
- **Run health:** all automated validation checks passed.
- **Selected model:** `mf` (selection rule: maximize validation `Recall@20`).
- **Business takeaway:** MF is currently strongest on offline retrieval quality; two-tower is improving but not yet the default.
- **Scope:** offline ranking quality only (not causal commercial lift).

## 2) Evaluation Setup

- Pipeline run: `src/mle_marketplace_growth/recommender/run_pipeline.py`
- Run used standard recommender contract from `docs/recommender/spec.md` (split/method/selection/metrics).

From `artifacts/recommender/as_of=<recommender_max_event_date>/train_metrics.json`:
- embedding dim: `64`
- epochs: `12`
- learning rate: `0.003`
- negative samples: `8`
- batch size: `4096`
- early-stop metric: `val_recall_at_k` (`K=20`, tolerance `0.0001`, rounds `4`)
- temperature: `0.7`
- normalize embeddings: `true` (cosine-style scoring)
- L2 reg: `0.0001`
- MF components: `64`

Benchmark note:
- `popularity` and `mf` are baseline comparators for sanity-checking retrieval quality.
- In this demo they are not fully tuned via wide hyperparameter search.

## 3) Offline Retrieval Metrics

How to read model comparison:
- All three models are tested on the same users and same split.
- `Recall@K` (Item Coverage): % of relevant items retrieved in top-K (primary selection uses `Recall@20`).
- `NDCG@K`(Item Relevance):: position-aware ranking quality in top-K.
- `HitRate@K` (Forgiving Item Coverage): % of users >= 1 relevant item in top-K (coverage-sanity guardrail).
- Higher metric values are better.
- The selected model is the one with highest validation `Recall@20`.

### Validation

| Model | Recall@10 | NDCG@10 | HitRate@10 | Recall@20 | NDCG@20 | HitRate@20 |
|---|---:|---:|---:|---:|---:|---:|
| popularity | 0.036948 | 0.017600 | 0.036948 | 0.056382 | 0.022610 | 0.056382 |
| mf | 0.083013 | 0.044036 | 0.083013 | 0.125000 | 0.054625 | 0.125000 |
| two_tower | 0.067658 | 0.036133 | 0.067658 | 0.099568 | 0.044201 | 0.099568 |

Validation readout (Recall@20):
- `mf` wins (`0.125000`)
- vs `popularity`: `+0.068618`
- vs `two_tower`: `+0.025432`

### Test

| Model | Recall@10 | NDCG@10 | HitRate@10 | Recall@20 | NDCG@20 | HitRate@20 |
|---|---:|---:|---:|---:|---:|---:|
| popularity | 0.049916 | 0.024529 | 0.049916 | 0.067195 | 0.029049 | 0.067195 |
| mf | 0.064795 | 0.033182 | 0.064795 | 0.095512 | 0.040882 | 0.095512 |
| two_tower | 0.045356 | 0.022635 | 0.045356 | 0.072714 | 0.029535 | 0.072714 |

Test readout (Recall@20):
- `mf` remains best (`0.095512`)
- vs `popularity`: `+0.028317`
- vs `two_tower`: `+0.022798`

Metric consistency check:
- Secondary metrics (NDCG, HitRate) show no conflict with primary `Recall@20` selection.
- Supporting evidence (test):
  - `Recall@20`: `mf 0.095512 > two_tower 0.072714 > popularity 0.067195`
  - `NDCG@20`: `mf 0.040882 > two_tower 0.029535 > popularity 0.029049`
  - `HitRate@20`: `mf 0.095512 > two_tower 0.072714 > popularity 0.067195`
- Interpretation: model ordering is directionally consistent across primary and guardrail metrics, so selecting `mf` by validation `Recall@20` is stable under offline evaluation.

## 4) Serving Outputs and Artifact Health

Key outputs present:
- `artifacts/recommender/as_of=<recommender_max_event_date>/topk_recommendations.csv`
- `artifacts/recommender/as_of=<recommender_max_event_date>/selected_model_meta.json`
- `artifacts/recommender/as_of=<recommender_max_event_date>/shared_context.json`
- `artifacts/recommender/as_of=<recommender_max_event_date>/models/<selected_model_name>/...`
- `artifacts/recommender/as_of=<recommender_max_event_date>/ann_index.bin`
- `artifacts/recommender/as_of=<recommender_max_event_date>/ann_index_meta.json`
- `artifacts/recommender/as_of=<recommender_max_event_date>/output_validation_summary.json`
- `artifacts/recommender/as_of=<recommender_max_event_date>/output_interpretation.md`

Validation summary status:
- overall `passed=true`
- selected model valid and present
- metric bounds checks passed
- recommendation output non-empty
- ANN artifacts present and consistent with selected model

## 5) Decision and Next Run Rule

- Current default candidate: `mf`.
- Keep ranking by offline `Recall@20` for structural model selection consistency.
- Keep quickstart workflow as source-of-truth reproducibility path.
- Two-tower architecture: keep the simpler embedding-only towers as the repo default; extra hidden-layer tower variants are out of scope for the current implementation.

## 6) Plots (Optional)
One model comparison bar chart for `Recall@20` (validation and test) across `popularity`, `mf`, `two_tower`.

![Recommender Recall@20 Comparison](../../artifacts/recommender/report_assets/model_recall_at20_comparison.png)
