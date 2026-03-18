# MLE Marketplace Growth Platform

An machine learning project for marketplace revenue growth via two complementary ML levers:

- Purchase Propensity Engine Flow (propensity scoring + budget-constrained incentive allocation)
- Recommender Engine Flow (candidate retrieval for personalized product exposure)

## Start Here (Architecture First)

`docs/architecture.pptx` is the front-facing architecture source of truth for this project.
Read it first for system flow, design intent, and implementation boundaries before running engine quickstarts.

## Problem Statement

Marketplaces often lose revenue in two ways:

- Incentives are distributed without strong user-level targeting.
- Product discovery is weak, so relevant items are not surfaced early.

The result is lower incremental revenue and inefficient budget use.

## How This Project Addresses It

| Component | What It Does | KPI Intent |
|---|---|---|
| Purchase Propensity Engine Flow | Predicts near-term purchase likelihood, ranks users by expected value, and compares policies offline under budget constraints. | Improve revenue per targeted user and budget efficiency. |
| Recommender Engine Flow | Learns user-item affinity and retrieves personalized product candidates. | Improve retrieval relevance and downstream engagement potential. |
| Shared Feature Layer | Enforces point-in-time features and time-based evaluation conventions for both engines. | Reduce leakage risk and improve reproducibility. |

## Getting Started

Environment setup (one-time):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Use engine-specific runbooks:
  - `docs/purchase_propensity/quickstart.md`
  - `docs/recommender/quickstart.md`
- Engine contracts (split/model/artifact/acceptance): `docs/purchase_propensity/spec.md`, `docs/recommender/spec.md`.
- Generated artifacts are intentionally committed in this repo as demo evidence; quickstarts and specs define the canonical artifact layouts.

## Tech Stack

| Scope | Stack |
|---|---|
| Shared | Python, DuckDB, NumPy, CSV artifacts, YAML config |
| Purchase propensity | scikit-learn, XGBoost |
| Recommender | scikit-learn (MF baseline), PyTorch (two-tower), FAISS (ANN retrieval) |

## Portfolio Signal Snapshot

| Engine | Signal Highlights |
|---|---|
| Purchase Propensity Engine Flow | Reproducible flow (feature store -> strict `10/1/1` split -> training -> budget-constrained backtest -> report), frozen structural decisions after initial sensitivity, ML policy benchmarked vs Random/RFM. |
| Recommender Engine Flow | Reproducible retrieval flow (shared feature layer -> purchase-invoice holdout -> Popularity/MF/Two-Tower training -> exact offline item-ranking eval -> FAISS ANN serving artifacts -> report), validation-driven model selection, Stage 1 candidate generation scope. |

## Key Limitations

| Limitation | Why It Matters |
|---|---|
| No online A/B testing | Repo is local/offline with no live traffic, so online impact cannot be measured directly. |
| No offline causal incrementality estimation for promotions | Dataset lacks randomized treatment assignment and reliable promo-exposure logs. |
| No recommender Stage 2 re-ranking | Current recommender scope is intentionally Stage 1 retrieval only. |
| Production next step (out of scope) | Run randomized A/B tests for promotion decisions before broad rollout. |

Policy and evaluation definitions live in engine specs to avoid doc duplication:
- `docs/purchase_propensity/spec.md`
- `docs/recommender/spec.md`

## Docs

- Architecture source of truth: `docs/architecture.pptx`
- Documentation map and conventions: `docs/README.md`

## Dataset

[Base dataset: Online Retail II (UCI transactional dataset)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data)

This Online Retail II dataset contains transactions from a UK-based non-store online retail business between 2009-12-01 and 2011-12-09. The company mainly sells unique all-occasion giftware, and many customers are wholesalers.
