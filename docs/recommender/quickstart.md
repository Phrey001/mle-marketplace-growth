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
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build --shared-config configs/shared.yaml --build-engines shared

# Fixed small-grid tuning sweep (heavier than main run)
PYTHONPATH=src python scripts/tune_recommender_minimal.py --config configs/recommender/default.yaml
```

## Recommended Run

```bash
# Shared layer (run once; reuse across engines)
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build --shared-config configs/shared.yaml --build-engines shared

# Recommender pipeline (feature-store gold -> train -> predict -> validate)
PYTHONPATH=src python -m mle_marketplace_growth.recommender.run_pipeline --config configs/recommender/default.yaml

# Regenerate Recall@20 comparison chart used in analysis report
PYTHONPATH=src python scripts/report_recommender_recall_chart.py
```

This command is prescribed and fail-fast:
- no optional paths in the default workflow (fixed time window + ANN-required serving).
- missing dependencies still fail immediately during module import/execution.
- engine-specific gold build requires prebuilt shared silver; run `--build-engines shared` first.
- recommender event-date bounds are validated against available shared silver event-date bounds during feature-store build.

This runs:
- feature-store build for recommender tables
- training/evaluation for popularity, MF, and two-tower retrieval
- Top-K candidate generation
- automated output validation + interpretation (includes random baseline anchor `K/N` and lift-vs-random framing)

## Key Config Params

| Group | Params |
|---|---|
| Split/versioning | `split_version` |
| Two-tower training | `embedding_dim`, `epochs`, `learning_rate`, `negative_samples`, `batch_size`, `l2_reg`, `max_grad_norm` |
| Two-tower convergence | `early_stop_rounds`, `early_stop_metric`, `early_stop_k`, `early_stop_tolerance` |
| Two-tower scoring/stability | `temperature`, `normalize_embeddings` |
| Optional tower depth | `tower_hidden_dim`, `tower_dropout` (`tower_hidden_dim=0` disables MLP towers) |
| Device | `device` (`auto` only: uses `cuda` when available, else `cpu`) |
| MF baseline | `mf_components`, `mf_n_iter`, `mf_weighting`, `mf_algorithm`, `mf_tol` |
| Popularity baseline | `popularity_transform` |
| Serving output | `top_k` (candidates written per user) |
| Offline eval cutoffs | `top_ks` (for example `10,20` computes Recall/NDCG/HitRate at K=10 and K=20) |

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
