# Recommender — Spec

High-level architecture lives in `docs/architecture.pptx`.
This file is the design contract for the recommender system.

## What This System Does

- Ranks item candidates for each user from implicit historical interactions
- Compares a popularity baseline, matrix factorization, and a two-tower neural retriever
- Selects the best model using validation retrieval quality
- Keeps offline evaluation exact while using ANN-style retrieval only for serving artifacts

## Objective

Build a Stage-1 retrieval engine that outputs personalized Top-K item candidates from implicit interactions.

## Scope

- In scope: candidate retrieval and offline evaluation.
- Out of scope: re-ranking, online serving stack, online A/B experimentation.
- Scope note: offline ranking metrics are not causal business-lift evidence.

## Core Contract

Non-negotiable behavior rules:

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
- offline evaluation outputs
- serving outputs
- validation/report outputs

Datetime ownership/bounds:
- Shared source data defines the maximum available event-date range.
- The recommender system chooses its own analysis window within that shared range.
- That window may be narrower than shared data availability, but not wider.
- The recommender feature store keeps only the latest canonical build; reruns overwrite prior outputs.

## System Flow

High-level lifecycle only:

1. Build the user-item interaction view and chronological split assignments.
2. Train and compare the candidate retrievers offline.
3. Select the best model by validation `Recall@20`.
4. Materialize serving-style retrieval artifacts for the selected model.
5. Validate the run outputs and generate interpretation text.

## Model Contract

Model and evaluation rules:

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
- Training is model-specific, while offline evaluation and prediction reuse a shared scoring contract.
- The current engine optimizes for discovery recommendations only; repeat-purchase / buy-again flows would be a separate recommendation objective.

Interaction signal:
- Implemented models use binary user-item interactions: each unique `(user, item)` pair is counted once (duplicates collapsed; quantity ignored).
- This is the standard implicit-feedback baseline used in many recommender systems.

Model-specific interaction treatment:

| Model | Interaction treatment |
|---|---|
| Popularity | Log-scaled counts of unique users interacting with each item |
| MF | TF-IDF weighted interaction matrix |
| Two-tower | Binary positive user-item pairs directly |

Retained for possible future extensions, but out of scope in the current repo:

| Future extension | Meaning |
|---|---|
| Count-weighted interactions | Repeated interactions increase signal strength; quantity is used |
| Log-scaled interaction strength | Large purchase quantities are damped before modeling |
| Revenue-aware recommendation | High-value transactions influence recommendation priority |

## Output Contract

Output categories only; exact commands live in the recommender quickstart.

Offline evaluation outputs:
- training summary
- validation retrieval metrics
- test retrieval metrics
- selected-model metadata
- shared runtime context
- selected-model exported artifacts

Serving outputs:
- item embedding state for retrieval
- ANN retrieval state and metadata
- Top-K recommendations

Validation/report outputs:
- validation summary
- interpretation markdown

## Acceptance Criteria

- Validation summary passes.
- Validation/test retrieval metrics include all three models.
- Selected model and selection rule are explicit in the run artifacts.
- ANN artifacts are present and consistent with the selected model.
- Top-K recommendations are non-empty and schema-valid.
