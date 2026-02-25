# Recommender Engine — Implementation Spec

High-level architecture lives in `docs/architecture.pptx`.
This document is the implementation contract for the recommender engine in this repository.

## Objective

Build an offline-evaluated retrieval engine that generates personalized Top-K item candidates per user from implicit transaction interactions.

Primary business intent:
- improve relevance of surfaced products
- increase downstream conversion opportunity from better candidate generation

## Scope Boundary

- Scope in this repo: retrieval/candidate generation stage only.
- Out of scope: feature-rich post-retrieval ranking model, online serving stack, and online A/B experimentation.
- Evaluation is offline only; no causal impact claim.

## Inputs and Data Model

Primary source table (already materialized by feature-store build):
- `data/silver/transactions_line_items/transactions_line_items.csv`

Expected interaction grain:
- one row per `(user_id, stock_code, event_timestamp)` after silver cleaning
- implicit feedback signal from purchase behavior (positive quantity transactions)

## Split Contract (Chronological, Leakage-Safe)

Per-user chronological split:
- Train interactions: all but latest two interactions
- Validation interaction: second-latest interaction
- Test interaction: latest interaction

Rules:
- Users with fewer than 3 valid interactions are excluded from offline ranking metrics.
- Item universe for candidate generation is defined from train interactions only.
- Validation/test interactions must not leak into train candidate scoring features.

## Baselines (Required)

1. Popularity baseline
- Rank items by global train interaction count.

2. Matrix factorization baseline
- Implicit-feedback MF using user-item interaction matrix.

## Primary Model (Required)

Two-tower retrieval model (minimal implementation):
- User tower input: user ID
- Item tower input: item ID
- User/item embeddings projected to same latent dimension
- Score = dot product between user and item embeddings

Training:
- implicit positive pairs from train interactions
- sampled negatives
- binary cross-entropy / sampled softmax style objective (implementation choice documented in code)

## Offline Evaluation

Metrics at `K`:
- Recall@K (primary model-selection metric)
- NDCG@K
- HitRate@K
- MRR@K (optional but recommended)

Required `K` values:
- `K = 10, 20`

Model selection rule:
- maximize validation `Recall@20`

Final reporting:
- report selected-model performance on test split
- compare selected model vs popularity and MF baselines

## Artifact Contract (Planned)

Feature-store outputs (already available):
- `data/gold/feature_store/recommender/interaction_events/interaction_events.csv`
- `data/gold/feature_store/recommender/user_item_splits/user_item_splits.csv`

Training/eval artifacts (to be implemented):
- `artifacts/recommender/train_metrics.json`
- `artifacts/recommender/validation_retrieval_metrics.json`
- `artifacts/recommender/test_retrieval_metrics.json`
- `artifacts/recommender/topk_recommendations.csv` (user-level candidates for selected model)
- `artifacts/recommender/output_validation_summary.json`
- `artifacts/recommender/output_interpretation.md`

## CLI Flow (Planned)

1. Build feature-store recommender datasets:
- `python -m mle_marketplace_growth.feature_store.build --build-engines recommender ...`

2. Train/evaluate recommender retrieval models:
- `python -m mle_marketplace_growth.recommender.train ...`

3. Generate candidate recommendations:
- `python -m mle_marketplace_growth.recommender.predict ...`

4. Validate and summarize outputs:
- `python -m mle_marketplace_growth.recommender.validate_outputs ...`

## Minimal MVP Implementation Plan

1. `src/mle_marketplace_growth/recommender/train.py`
- load interaction_events + user_item_splits
- train popularity + MF + two-tower
- compute validation/test metrics@K
- save model artifacts + metrics JSON

2. `src/mle_marketplace_growth/recommender/predict.py`
- load selected model
- emit user-level Top-K candidates CSV

3. `src/mle_marketplace_growth/recommender/validate_outputs.py`
- sanity checks for required artifacts/metric ranges
- write interpretation markdown

4. `src/mle_marketplace_growth/recommender/run_pipeline.py`
- one-command orchestration analogous to purchase propensity flow

## Acceptance Criteria

- All required artifacts are generated and schema-valid.
- Validation/test metrics include all required baselines and selected model.
- Selected model is explicitly recorded with selection rule (`Recall@20` on validation).
- Top-K output exists for at least all eligible test users.
- Output validation summary passes without failures.
