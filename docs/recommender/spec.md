# Recommender Engine — Spec

High-level architecture lives in `docs/architecture.pptx`.
This document is the implementation contract for the recommender engine in this repository.

## Spec Lifecycle

- This spec starts as a preliminary design document derived from `docs/architecture.pptx`.
- As code and run artifacts stabilize, it is finalized as the implementation contract for this repo.

## Objective

Build an offline-evaluated Stage-1 retrieval engine that generates personalized Top-K item candidates from implicit transaction interactions.

## Scope and Boundary

- In scope: retrieval/candidate generation and offline evaluation.
- Out of scope: downstream re-ranking model, online serving stack, and online A/B experimentation.
- Scope note: offline ranking metrics are not causal business-lift evidence.

## Tech Stack

- Feature/data layer: DuckDB SQL + CSV materialization.
- Baselines: NumPy + scikit-learn (`TruncatedSVD` for MF).
- Two-tower model: PyTorch.
- Retrieval index: FAISS HNSW (inner-product ANN), fail-fast required.

## Inputs and Feature Store Contract

Required feature-store inputs:
- `data/gold/feature_store/recommender/interaction_events/interaction_events.csv`
- `data/gold/feature_store/recommender/user_item_splits/user_item_splits.csv`
- `data/gold/feature_store/recommender/user_index/user_index.csv`
- `data/gold/feature_store/recommender/item_index/item_index.csv`

Split contract (chronological, leakage-safe):
- `train`: all but latest two interactions per user
- `val`: second-latest interaction per user
- `test`: latest interaction per user
- users with fewer than 3 interactions are excluded from ranking metrics
- train item universe defines retrieval candidates

## Models

| Level | Model | Role |
|---|---|---|
| 0 | Random | Theoretical floor (`K/N` anchor in interpretation) |
| 1 | Popularity | Strong heuristic baseline |
| 2 | Matrix Factorization (MF) | Classical ML baseline |
| 3 | Two-tower | Neural retrieval baseline (primary model) |

Training characteristics:
- implicit positives from train interactions
- sampled negatives
- binary objective for pair discrimination

## Pipeline Map

| Stage | Script | Key output(s) |
|---|---|---|
| Feature-store build | `mle_marketplace_growth.feature_store.build_gold_recommender` | `interaction_events.csv`, `user_item_splits.csv`, index tables |
| Model training/evaluation | `mle_marketplace_growth.recommender.train` | `train_metrics.json`, `validation_retrieval_metrics.json`, `test_retrieval_metrics.json`, `model_bundle.pkl` |
| Retrieval serving artifacts | `mle_marketplace_growth.recommender.predict` | `item_embeddings.npy`, `ann_index.bin`, `ann_index_meta.json`, `topk_recommendations.csv` |
| Output validation/reporting | `mle_marketplace_growth.recommender.validate_outputs` | `output_validation_summary.json`, `output_interpretation.md` |

## Serving (Retrieval)

- Precompute item embeddings from selected model.
- Build ANN index over item embeddings (FAISS HNSW IP).
- Produce user-level Top-K recommendations from ANN retrieval.
- Fail fast when ANN dependency/artifacts are unavailable.

## Offline Evaluation

Required metrics at K:
- Recall@K (primary selection metric)
- NDCG@K
- HitRate@K

Required K values:
- `K = 10, 20`

Selection rule:
- maximize validation `Recall@20`

Reporting:
- compare `popularity`, `mf`, `two_tower` on validation and test
- record selected model and selection rule in artifacts

## Artifact Contract

Training/evaluation artifacts:
- `artifacts/recommender/train_metrics.json`
- `artifacts/recommender/validation_retrieval_metrics.json`
- `artifacts/recommender/test_retrieval_metrics.json`
- `artifacts/recommender/model_bundle.pkl`

Serving/retrieval artifacts:
- `artifacts/recommender/item_embeddings.npy`
- `artifacts/recommender/item_embedding_index.json`
- `artifacts/recommender/ann_index.bin`
- `artifacts/recommender/ann_index_meta.json`
- `artifacts/recommender/topk_recommendations.csv`

Validation/report artifacts:
- `artifacts/recommender/output_validation_summary.json`
- `artifacts/recommender/output_interpretation.md`
- `docs/recommender/analysis_report.md`

Run commands are documented in `docs/recommender/quickstart.md`.

## Acceptance Criteria

- Output validation summary passes.
- Validation/test metrics include all three models (`popularity`, `mf`, `two_tower`).
- Selected model and selection rule are explicitly recorded.
- ANN artifacts are present and consistent with selected model.
- Top-K output is non-empty and schema-valid.
