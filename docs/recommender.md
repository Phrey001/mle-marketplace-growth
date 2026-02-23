# Retrieval & Ranking Engine (Content/Product Selection)

What content/product to show to end users, implemented as a two-tower collaborative filtering retrieval model.

This document is aligned to `docs/architecture.pptx`.

---

## Engine Flow

- **Preprocessing in Feature Store:**
  - Interaction matrix construction (user–item interactions with time-based data)
  - Time-based split (train/val/test)
  - Precomputed item embeddings (for fast retrieval)
- **Training Pipeline (Offline):** two-tower collaborative filtering (learn user + item embeddings)
- **Serving (Retrieval) Pipeline:** user–item dot product scoring
- **Offline Evaluation:** proxy metrics for commercial engagement (recall, NDCG, hit rate)
- **Periodic Retraining:** if running in a production environment

---

## Two-Tower Retrieval (Implementation Notes)

- **User tower:** user embedding
- **Item tower:** item embedding
- **Scoring:** dot product similarity
- **Retrieval workflow:** compute embeddings → score via dot product → retrieve top-K candidates

---

## Terms for Reference

- **Collaborative Filtering:** behaviour-driven learning from user–item interactions
- **Two-Tower Model Training:** learn user and item embeddings in a shared embedding space for recommendations

---

## Optional Baselines (Offline)

- Popularity ranking
- Matrix factorization (implicit feedback)

See also: `docs/recommender_spec.md` (supplemental implementation/spec notes).
