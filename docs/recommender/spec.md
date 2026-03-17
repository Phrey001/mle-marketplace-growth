# Recommender Engine — Spec

High-level architecture lives in `docs/architecture.pptx`.
This file is the implementation contract for the recommender engine.

## Objective

Build a Stage-1 retrieval engine that outputs personalized Top-K item candidates from implicit interactions.

## Scope

- In scope: candidate retrieval and offline evaluation.
- Out of scope: re-ranking, online serving stack, online A/B experimentation.
- Scope note: offline ranking metrics are not causal business-lift evidence.

## Core Contract

| Area | Contract |
|---|---|
| Split mode | User-level chronological holdout at invoice-moment grain: train=older invoice moments, val=second-latest invoice moment, test=latest invoice moment |
| Eligibility | Users with fewer than 3 interactions are excluded from ranking evaluation |
| Candidate universe | Retrieval candidates are from train item universe |
| Models compared | `popularity`, `mf`, `two_tower` |
| Selection rule | Maximize validation `Recall@20` |
| Required K | `K=20` |
| ANN backend | FAISS HNSW inner-product index, fail-fast if unavailable |

Datetime ownership/bounds:
- Shared silver data availability is defined by `configs/shared.yaml`.
- Engine datetime is owned by recommender config (`recommender_min_event_date`, `recommender_max_event_date`).
- Engine datetime may be narrower than shared bounds, but must not exceed shared silver event-date bounds (fail-fast on violation).
- Recommender feature store keeps only the latest canonical build (no experiment tracking); reruns overwrite prior outputs.

## Pipeline Map

| Stage | Script | Key output(s) |
|---|---|---|
| Feature-store build | `mle_marketplace_growth.feature_store.build_gold_recommender` | `interaction_events.parquet`, `user_item_splits.parquet`, `user_index.parquet`, `item_index.parquet` |
| Train/evaluate | `mle_marketplace_growth.recommender.train` | `train_metrics.json`, `validation_retrieval_metrics.json`, `test_retrieval_metrics.json`, `model_bundle.pkl` |
| Build retrieval artifacts | `mle_marketplace_growth.recommender.predict` | `item_embeddings.npy`, `item_embedding_index.json`, `ann_index.bin`, `ann_index_meta.json`, `topk_recommendations.csv` |
| Output validation/report text | `mle_marketplace_growth.recommender.validate_outputs` | `output_validation_summary.json`, `output_interpretation.md` |

## Model Contract

| Item | Contract |
|---|---|
| Baseline hierarchy | Random floor (`K/N`) -> Popularity -> MF -> Two-tower |
| MF implementation | `TruncatedSVD` over implicit interaction matrix |
| Two-tower architecture | Embedding-only user tower and item tower (no hidden MLP layers) |
| Two-tower objective | Contrastive cross-entropy with in-batch + sampled negatives |
| Two-tower scoring | L2-normalized dot product (cosine-style); temperature applied in training logits |
| Primary metric | `Recall@20` |
| Guardrails | `NDCG@20`, `HitRate@20` |

Design note:
- A deeper one-hidden-layer tower variant was considered earlier, but the current repo keeps embedding-only towers to avoid extra complexity without clear benefit for this demo implementation.

Interaction signal:
- Implemented models use binary user-item interactions: each unique `(user, item)` pair is counted once (duplicates collapsed; quantity ignored).
- This is the standard implicit-feedback baseline used in many recommender systems.
- Model-specific transformations are then applied to the interaction signals:
  - Popularity: log-scaled counts of unique users interacting with each item
  - MF: TF-IDF weighted interaction matrix
  - Two-tower: uses binary pairs directly
- Out of scope (dataset-specific): the feature-store gold tables still retain purchase quantity (`weight`) for potential future extensions:
  - count-weighted interactions: repeated interactions increase signal strength (retain repeated user-item interactions; no deduplication; quantity used)
  - log-scaled interaction strength: reduces the impact of extreme purchase quantities
  - revenue-aware recommendation: prioritizes high-value transactions

## Artifact Contract

Training/evaluation artifacts (`artifacts/recommender/as_of=<recommender_max_event_date>/`):

- `train_metrics.json`
- `validation_retrieval_metrics.json`
- `test_retrieval_metrics.json`
- `model_bundle.pkl`

Retrieval artifacts (`artifacts/recommender/as_of=<recommender_max_event_date>/`):

- `item_embeddings.npy`
- `item_embedding_index.json`
- `ann_index.bin`
- `ann_index_meta.json`
- `topk_recommendations.csv`

Validation/report artifacts (`artifacts/recommender/as_of=<recommender_max_event_date>/`):

- `output_validation_summary.json`
- `output_interpretation.md`

## Acceptance Criteria

- Output validation summary passes.
- Validation/test metric files include all three models.
- Selected model and selection rule are explicit in artifacts.
- ANN artifacts are present and consistent with selected model.
- Top-K recommendations are non-empty and schema-valid.

Run commands: `docs/recommender/quickstart.md`.
