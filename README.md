# MLE Marketplace Growth Platform

An end-to-end machine learning system for optimizing marketplace revenue.
It models two complementary levers—promotion/incentive allocation and personalized retrieval/ranking—on a shared data + feature pipeline.
Deeper technical details live in `docs/` (linked below).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Docs

- `docs/architecture.pptx` — source-of-truth architecture diagrams and flow.
- `docs/README.md` — documentation map and conventions.
- `docs/feature_store.md` — feature store layout + “as-of” rules (shared preprocessing layer).
- `docs/growth_uplift.md` — causal incentive allocation (uplift modeling + policy evaluation).
- `docs/growth_uplift_spec.md` — supplemental spec notes (assumptions, evaluation details).
- `docs/recommender.md` — retrieval & ranking engine (two-tower candidate generation + evaluation).
- `docs/recommender_spec.md` — supplemental spec notes (data split, baselines, training details).

## Business Framing

Marketplaces tend to increase revenue via two primary mechanisms:

- Incentive allocation: selectively distributing promotions to maximize incremental revenue.
- Personalization & ranking: retrieving and ranking items that align with user intent.

This repository implements both levers:

1. Growth Uplift Engine (Causal Incentive Allocation)
2. Retrieval & Ranking Recommender (Two-Tower Collaborative Filtering)

## Dataset

[Base dataset: Online Retail II (UCI transactional dataset)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data)

This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.

## System Architecture Overview

At a high level, the system flows from transactions dataset → shared feature engineering → two independent ML engines → offline evaluation/policy simulation.
See `docs/architecture.pptx` (source of truth) for detailed flows and boundaries.

## Feature Store (Shared Feature Engineering Layer)

Planned derived features include user behavioral features (e.g., RFM), time-based interaction splits, and revenue modeling inputs.

## Market Growth Levers

### Growth Uplift Engine (Growth Decisioning)

Optimizes voucher/promotion allocation under capacity constraints using causal uplift modeling and counterfactual policy evaluation.
See `docs/growth_uplift.md`.

### Retrieval & Ranking Engine (Content/Product Selection)

Generates personalized top-K item candidates using a two-tower retrieval model and compares against simple baselines offline.
See `docs/recommender.md`.

## Technical Stack

- Python
- PyTorch (two-tower retrieval)
- Pandas / NumPy
- Modular packaging under `src/`

## Design Principles

- Clear separation of recall vs ranking
- Revenue-aware evaluation
- Time-based validation (no leakage)
- Deterministic reproducibility
- Modular system design

## Key Capabilities Demonstrated

- Causal ML for incentive optimization
- Deep retrieval modeling (two-tower)
- Collaborative filtering baselines
- Counterfactual policy evaluation
- Business-aligned ML system design
