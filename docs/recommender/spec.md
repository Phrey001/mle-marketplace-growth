# Retrieval & Ranking Engine — Spec Notes (Supplemental)

This document captures implementation/spec details that may be useful for designing components and interfaces.
The high-level architecture and flows live in `docs/architecture.pptx`.

---

## Objective

Generate personalized top-K item recommendations/candidates to improve commercial outcomes (e.g., GMV).

---

## Data Preparation

- Construct user–item interaction matrix
- Time-based split:
  - Train: historical interactions
  - Validation: second-last interaction
  - Test: last interaction
- Implicit feedback setting

---

## Baselines (Offline)

1. Popularity ranking
2. Matrix factorization (implicit feedback)

---

## Two-Tower Retrieval Model (Example)

Architecture:

- User tower: user ID embedding
- Item tower: item ID embedding

Scoring:

- Dot product similarity

Training:

- Negative sampling
- Cross-entropy loss
- Mini-batch optimization (PyTorch)

---

## Retrieval Workflow

1. Compute user embeddings
2. Compute item embeddings
3. Score items via dot product
4. Retrieve top-K candidates

---

## Evaluation Metrics

- Recall@K
- NDCG@K
- Hit rate@K
- MRR@K (optional)

Comparison against baselines included.

---

## Business Interpretation

The two-tower model represents the candidate generation (recall) stage in a modern recommendation pipeline.
In production systems, this is typically followed by a feature-rich ranking model.

