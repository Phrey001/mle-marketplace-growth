# MLE Marketplace Growth Platform

An end-to-end machine learning project for marketplace revenue growth via two complementary ML levers:

- Growth uplift decisioning (promotion allocation)
- Retrieval/personalization (two-tower candidate generation)

## Problem Statement

Marketplaces often lose revenue in two ways:

- Promotions are over-distributed to users who would have purchased anyway.
- Product discovery is weak, so relevant items are not surfaced early.

The result is lower incremental revenue and inefficient budget use.

## How This Project Addresses It

- **Growth Uplift Decisioning:** estimates incremental impact of treatment and prioritizes users under budget/capacity constraints.
- **Retrieval/Personalization:** learns user-item affinity to generate better top-K candidates for downstream ranking.
- **Shared Feature Layer:** enforces point-in-time features and time-based evaluation to reduce leakage and improve reproducibility.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m mle_marketplace_growth.feature_store.build
```

The build command also runs feature-store DQ checks and writes a run manifest.

## Docs

- Architecture source of truth: `docs/architecture.pptx`
- Documentation map and conventions: `docs/README.md`

## Dataset

[Base dataset: Online Retail II (UCI transactional dataset)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data)

This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.
