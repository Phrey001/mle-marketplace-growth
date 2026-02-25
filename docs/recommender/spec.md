# Recommender Engine — Implementation Spec

High-level architecture lives in `docs/architecture.pptx`.
This document is the implementation contract for the recommender engine in this repository.

## Objective

Build an offline-evaluated retrieval engine that generates personalized Top-K item candidates per user from implicit transaction interactions.
This corresponds to Stage 1 candidate retrieval in a retrieval-and-ranking architecture.

Primary business intent:
- improve relevance of surfaced products
- increase downstream conversion opportunity from better candidate generation

## Scope Boundary

- Scope in this repo: retrieval/candidate generation stage is implemented; ranking stage is represented as a downstream interface contract (not a full feature-rich ranker implementation).
- Out of scope: feature-rich post-retrieval ranking model, online serving stack, and online A/B experimentation.
- Evaluation is offline only; no causal impact claim.

## Inputs and Data Model

Primary source table (already materialized by feature-store build):
- `data/silver/transactions_line_items/transactions_line_items.csv`

Expected interaction grain:
- one row per `(user_id, stock_code, event_timestamp)` after silver cleaning
- implicit feedback signal from purchase behavior (positive quantity transactions)

Feature-store preprocessing assumptions (slide-aligned):
- interaction matrix construction from time-aware user-item interaction features
- behavioural aggregates available (for example RFM-style features) for optional diagnostics or downstream ranking interfaces

## Split Contract (Chronological, Leakage-Safe)

Per-user chronological split:
- Train interactions: all but latest two interactions
- Validation interaction: second-latest interaction
- Test interaction: latest interaction

Rules:
- Users with fewer than 3 valid interactions are excluded from offline ranking metrics.
- Item universe for candidate generation is defined from train interactions only.
- Validation/test interactions must not leak into training features or candidate retrieval model fitting.

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

Serving contract (slide-aligned):
- precompute and persist item embeddings from the selected model
- compute user embeddings at scoring time
- retrieve Top-K candidates via user-item dot product
- support ANN-based nearest-neighbor retrieval for large catalogs (exact search fallback is acceptable for demo-scale catalogs)

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

Business interpretation note:
- these are proxy engagement/relevance metrics for retrieval quality; they are not direct causal commercial lift estimates

## Artifact Contract (Planned)

Feature-store outputs (already available):
- `data/gold/feature_store/recommender/interaction_events/interaction_events.csv`
- `data/gold/feature_store/recommender/user_item_splits/user_item_splits.csv`

Training/eval artifacts (to be implemented):
- `artifacts/recommender/train_metrics.json`
- `artifacts/recommender/validation_retrieval_metrics.json`
- `artifacts/recommender/test_retrieval_metrics.json`
- `artifacts/recommender/item_embeddings.npy` (precomputed item embeddings for retrieval serving)
- `artifacts/recommender/item_embedding_index.json` (item-id to embedding-row mapping)
- `artifacts/recommender/ann_index.bin` (optional ANN index for fast retrieval at larger scale)
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

5. (Optional downstream) Rank retrieved candidates with business/ranking layer:
- output contract includes user-level candidate list ready for ranking stage.

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
- Precomputed item embeddings and index mapping are materialized for retrieval serving.
- Output validation summary passes without failures.

## Retraining Cadence

- Production intent (slide-aligned): periodic retraining to refresh embeddings and interaction patterns.
- Demo recommendation: start with single-cycle implementation; add periodic retraining orchestration after baseline retrieval quality is stable.
