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
| Split mode | User-level chronological holdout at purchase-invoice grain: train=older purchase invoices, val=second-latest purchase invoice, test=latest purchase invoice |
| Eligibility | Users with fewer than 3 purchase invoices are excluded from ranking evaluation |
| Candidate universe | Retrieval candidates are from train item universe |
| Recommendation policy | Discovery-oriented: offline evaluation and serving exclude train-seen items for every model family |
| Offline eval retrieval | Exact ranking over the full item universe after excluding train-seen items for each user |
| Serving retrieval | ANN via FAISS HNSW inner-product search for efficient retrieval over the full item universe |
| Models compared | `popularity`, `mf`, `two_tower` |
| Selection rule | Maximize validation `Recall@20` |
| Required K | `K=20` |
| ANN backend | FAISS HNSW inner-product index, fail-fast if unavailable |

Artifact organization:
- Run outputs are grouped into three artifact categories:
  - offline evaluation artifacts
  - serving artifacts
  - validation/report artifacts
- Exact output paths and run commands live in `docs/recommender/quickstart.md`.

Datetime ownership/bounds:
- Shared silver data availability is defined by `configs/shared.yaml`.
- Engine datetime is owned by recommender config (`recommender_min_event_date`, `recommender_max_event_date`).
- Engine datetime may be narrower than shared bounds, but must not exceed shared silver event-date bounds (fail-fast on violation).
- Recommender feature store keeps only the latest canonical build (no experiment tracking); reruns overwrite prior outputs.

## Pipeline Map

| Stage | Script | Key output(s) |
|---|---|---|
| Feature-store build | `mle_marketplace_growth.feature_store.build_gold_recommender` | interaction events, user-item split assignments, user index, item index |
| Train/evaluate/select | `mle_marketplace_growth.recommender.train_and_select` | training summary, validation/test retrieval metrics, selected-model metadata, shared runtime context, selected-model exported artifacts |
| Build retrieval artifacts | `mle_marketplace_growth.recommender.predict` | item embedding matrix, item embedding index, ANN index + metadata, Top-K recommendations |
| Output validation/report text | `mle_marketplace_growth.recommender.validate_outputs` | validation summary, interpretation markdown |

## Model Contract

| Item | Contract |
|---|---|
| Baseline hierarchy | Random floor (`K/N`) -> Popularity -> MF -> Two-tower |
| MF implementation | `TruncatedSVD` over sparse implicit user-item matrix |
| Two-tower architecture | Embedding-only user tower and item tower (no hidden MLP layers) |
| Two-tower train input | Explicit positive `(user_idx, item_idx)` pairs plus validation cache for early stopping |
| Two-tower objective | Contrastive cross-entropy with in-batch + sampled negatives |
| Two-tower scoring | L2-normalized dot product (cosine-style); temperature applied in training logits |
| Primary metric | `Recall@20` |
| Guardrails | `NDCG@20`, `HitRate@20` |

Prediction/evaluation unit:
- Models rank item IDs, not invoice IDs.
- Held-out truth is the item set from the user's held-out purchase invoice (next basket).

Design note:
- A deeper one-hidden-layer tower variant was considered earlier, but the current repo keeps embedding-only towers to avoid extra complexity without clear benefit for this demo implementation.
- Training and scoring are already split by recommender model family, while shared evaluation and prediction load the selected model through a common scorer contract.
- Offline evaluation belongs to the `train_and_select` stage. `helpers/metrics.py` provides the ranking/metric primitives, not a separate pipeline stage.
- The current engine optimizes for discovery recommendations only; repeat-purchase / buy-again flows would be a separate recommendation objective.

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

Offline evaluation artifacts:
- training summary
- validation retrieval metrics
- test retrieval metrics
- selected-model metadata
- shared runtime context
- selected-model exported artifacts

Serving artifacts:
- item embedding matrix
- item embedding index
- ANN index and metadata
- Top-K recommendations

Validation/report artifacts:
- validation summary
- interpretation markdown

## Acceptance Criteria

- Validation summary passes.
- Validation/test retrieval metrics include all three models.
- Selected model and selection rule are explicit in the run artifacts.
- ANN artifacts are present and consistent with the selected model.
- Top-K recommendations are non-empty and schema-valid.

Run commands: `docs/recommender/quickstart.md`.
