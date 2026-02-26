# Recommender Quickstart

Environment setup is shared at repository root `README.md`.

## Recommended Run

```bash
PYTHONPATH=src python -m mle_marketplace_growth.recommender.run_pipeline --config configs/recommender/default.yaml
```

This command is prescribed and fail-fast:
- no optional paths in the default workflow (fixed time window + ANN-required serving).
- missing dependencies still fail immediately during module import/execution.

This runs:
- feature-store build for recommender tables
- training/evaluation for popularity, MF, and two-tower retrieval
- Top-K candidate generation
- automated output validation + interpretation

## Key Config Params

- `split_version`: split strategy ID (tracks which train/val/test split rule was used).
- `top_k`: number of candidates written per user in serving output (`topk_recommendations.csv`).
- `top_ks`: metric cutoffs used in offline evaluation (for example `10,20` computes Recall/NDCG/HitRate at K=10 and K=20).

## Outputs To Review

- `artifacts/recommender/output_validation_summary.json`
- `artifacts/recommender/output_interpretation.md`
- `artifacts/recommender/validation_retrieval_metrics.json`
- `artifacts/recommender/test_retrieval_metrics.json`
- `artifacts/recommender/topk_recommendations.csv`

## Tests

Unit tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_recommender_minimal.py
```

Design/spec reference:
- `docs/recommender/spec.md`
