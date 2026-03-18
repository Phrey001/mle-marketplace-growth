# Recommender — Analytical Report

## Report Metadata

- **Artifact source:** `artifacts/recommender/as_of=<recommender_max_event_date>/`
- **Run scope:** main recommender pipeline plus two-tower tuning sweep
- **Update policy:** refresh this report after reruns with material metric, selection, or tuning changes

## 1) Executive Summary

| Item | Current readout |
|---|---|
| Selected model | `mf` |
| Selection rule | maximize validation `Recall@20` |
| Validation winner | `mf` at `0.052540` |
| Test winner | `mf` at `0.043009` |
| Best two-tower tuning trial | `trial_1` (`temperature=0.5`) |
| Validation health | passed |

- **Decision:** keep `mf` as the current retrieval default.
- **Takeaway:** MF remains the strongest offline retrieval model. Two-tower beats popularity on recall and NDCG, but it still trails MF by a clear margin.
- **Scope:** offline retrieval quality only; this report does not claim causal business lift.

## 2) Evaluation Setup

- Pipeline entrypoint: `src/mle_marketplace_growth/recommender/run_pipeline.py`
- Spec contract: `docs/recommender/spec.md`
- Split rule: user-level chronological holdout at purchase-invoice grain
- Prediction unit: ranked item IDs
- Held-out truth unit: the item set from the user's held-out purchase invoice
- Recommendation objective: discovery only
- Offline candidate universe: full item universe after excluding train-seen items
- Serving retrieval: FAISS HNSW ANN over item embeddings
- Selection metric: validation `Recall@20`
- Eligible users scored in current run: `1953`

Main run config from `artifacts/recommender/as_of=<recommender_max_event_date>/offline_eval/train_metrics.json`:

| Model | Key settings |
|---|---|
| `mf` | `components=64`, `n_iter=15`, `weighting=tfidf` |
| `two_tower` | `embedding_dim=64`, `epochs=12`, `lr=0.003`, `negatives=8`, `batch_size=4096`, `l2=0.0001`, `max_grad_norm=1.0`, `early_stop_rounds=4`, `early_stop_k=20`, `early_stop_tolerance=0.0001`, `temperature=0.7` |

Benchmark note:
- `popularity` and `mf` are baseline comparators.
- The optional tuning sweep only changes two-tower hyperparameters; MF and popularity stay fixed.

## 3) Offline Retrieval Metrics

How to read model comparison:
- `Recall@20`: primary selection metric; fraction of held-out relevant items recovered in top-20.
- `NDCG@20`: rank-sensitive quality; earlier relevant hits count more.
- `HitRate@20`: fraction of users with at least one relevant item in the top-20.
- Higher is better.

### Validation

| Model | Recall@20 | NDCG@20 | HitRate@20 |
|---|---:|---:|---:|
| popularity | 0.028270 | 0.034973 | 0.310292 |
| mf | 0.052540 | 0.063791 | 0.448541 |
| two_tower | 0.036484 | 0.040071 | 0.295955 |

Validation readout:
- Winner: `mf`
- Margin vs `popularity`: `+0.024270`
- Margin vs `two_tower`: `+0.016056`

### Test

| Model | Recall@20 | NDCG@20 | HitRate@20 |
|---|---:|---:|---:|
| popularity | 0.022910 | 0.027879 | 0.261649 |
| mf | 0.043009 | 0.052643 | 0.399386 |
| two_tower | 0.028361 | 0.032184 | 0.254480 |

Test readout:
- Winner: `mf`
- Margin vs `popularity`: `+0.020099`
- Margin vs `two_tower`: `+0.014648`

Metric consistency check:
- `Recall@20`: `mf > two_tower > popularity`
- `NDCG@20`: `mf > two_tower > popularity`
- `HitRate@20`: `mf > popularity > two_tower`
- Interpretation:
  - MF is clearly strongest.
  - Two-tower improves on recall and NDCG versus popularity, but not on hit rate.
  - No metric suggests overturning the validation-based `mf` selection.

## 4) Tuning Sweep Readout

Source:
- `artifacts/recommender/tuning/tuning_summary.json`

| Item | Readout |
|---|---|
| Sweep strategy | `fixed_small_grid_two_tower_local` |
| Best overall trial | `trial_default` |
| Best overall selected model | `mf` |
| Best overall validation `Recall@20` | `0.052540` |
| Best overall test `Recall@20` | `0.043009` |
| Best two-tower trial | `trial_1` |
| Best two-tower override | `temperature=0.5` |
| Best two-tower validation `Recall@20` | `0.038492` |
| Best two-tower test `Recall@20` | `0.028924` |

Tuning takeaway:
- The tuning sweep did not change the overall winner; MF still dominates the leaderboard.
- Lowering two-tower temperature from `0.7` to `0.5` produced the best two-tower result in this sweep.
- The best tuned two-tower run improves modestly over the main-pipeline two-tower result, but the gain is too small to threaten MF.

## 5) Serving Outputs and Artifact Health

Key outputs present:
- `offline_eval/selected_model_meta.json`
- `offline_eval/shared_context.json`
- `offline_eval/models/<selected_model_name>/...`
- `serving/topk_recommendations.csv`
- `serving/ann_index.bin`
- `serving/ann_index_meta.json`
- `report/output_validation_summary.json`
- `report/output_interpretation.md`

Validation summary status:
- overall `passed=true`
- selected model valid and present
- validation/test metric rows present for all three models
- metric bounds checks passed
- recommendation output non-empty
- ANN artifacts present and aligned with the selected model

## 6) Decision and Next Run Rule

- Current default candidate: `mf`
- Keep selecting by validation `Recall@20`
- Keep `popularity` as the sanity baseline
- Keep two-tower as the challenger model family
- If two-tower is revisited, the most defensible next step is richer feature-store inputs rather than more cosmetic model churn

## 7) Plot

Recall@20 comparison chart:

![Recommender Recall@20 Comparison](../../artifacts/recommender/report_assets/model_recall_at20_comparison.png)
