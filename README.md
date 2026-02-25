# MLE Marketplace Growth Platform

An end-to-end machine learning project for marketplace revenue growth via two complementary ML levers:

- Purchase propensity scoring + incentive allocation
- Personalized product candidate generation

## Problem Statement

Marketplaces often lose revenue in two ways:

- Incentives are distributed without strong user-level targeting.
- Product discovery is weak, so relevant items are not surfaced early.

The result is lower incremental revenue and inefficient budget use.

## How This Project Addresses It

- **Purchase Propensity:** predicts near-term purchase likelihood, ranks users by expected value under budget constraints, and compares targeting policies offline.
- **Personalized Retrieval:** learns user-item affinity to surface more relevant candidate products.
- **Shared Feature Layer:** enforces point-in-time features and time-based evaluation to reduce leakage and improve reproducibility.
- **Business KPI Intent:** improve revenue per incentive dollar and improve relevance of surfaced items.

## Getting Started

- Run setup, recommended commands, output checks, tests, and automated interpretation review from `docs/quickstart.md`.
- Modeling choices (scaling, spend capping, calibration) are documented in `docs/purchase_propensity/spec.md`.
- Window sensitivity now evaluates materialized feature-lookback profiles (`60/90/120`) for model-design comparison; main pipeline default remains `30d` target + `90d` lookback.

## Key Limitations

- **Not included:** online A/B testing. Why: this repo is local/offline and has no live traffic.
- **Not included:** offline causal incrementality estimation for promotions (for example, “did the promo itself cause extra purchases?”). Why: no randomized treatment assignment or reliable promo-exposure logs in this dataset.
- **Production recommendation (out of scope for this repo):** run randomized A/B testing for promotion decisions before broad rollout to measure true incremental impact.

## Policy Interpretation

- The 3-policy comparison is implemented: ML expected-value targeting, random baseline, and RFM heuristic baseline.
- Budget is used to decide **who to target** (allocation logic), not to simulate behavior change from incentive exposure.
- Results compare policy performance on historical holdout outcomes; they do **not** estimate causal incrementality.
- Policy definitions and design details are documented in `docs/purchase_propensity/spec.md`.

## Docs

- Architecture source of truth: `docs/architecture.pptx`
- Documentation map and conventions: `docs/README.md`

## Dataset

[Base dataset: Online Retail II (UCI transactional dataset)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data)

This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.
