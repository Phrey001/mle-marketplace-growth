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
```

Recommended uplift run (use an earlier snapshot date so future labels are non-empty):

```bash
# 1) Build feature store at an earlier as_of_date for non-empty uplift label window
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build --as-of-date 2011-11-09

# 2) Train uplift model on matching uplift_train_dataset snapshot
PYTHONPATH=src python -m mle_marketplace_growth.growth_uplift.train --input-csv data/gold/feature_store/growth_uplift/uplift_train_dataset/as_of_date=2011-11-09/uplift_train_dataset.csv

# 3) Score matching user_features snapshot, then evaluate policy lift
PYTHONPATH=src python -m mle_marketplace_growth.growth_uplift.predict --input-csv data/gold/feature_store/growth_uplift/user_features_asof/as_of_date=2011-11-09/user_features_asof.csv
PYTHONPATH=src python -m mle_marketplace_growth.growth_uplift.evaluate

# 4) Run budget-constrained policy simulation from response proxies
PYTHONPATH=src python -m mle_marketplace_growth.growth_uplift.policy_simulation
```

Review `artifacts/growth_uplift/evaluation.json` (`sanity`, `interpretation_kpis`) and `artifacts/growth_uplift/evaluation_policy_lift.png` for high-level policy quality.
If `sanity.flags` includes `all_outcomes_zero_check_as_of_date_window`, rerun with an earlier `--as-of-date`.
Check `artifacts/growth_uplift/train_metrics.json` for `selected_model_name` and candidate model comparison.
Current uplift evaluation is simulation-only (synthetic treatment assignment) and should be interpreted as model-logic validation, not direct business-impact evidence.
For business-impact claims, use real treatment logs (or a fully synthetic sandbox where outcomes are also simulated from explicit promotion assumptions).
Policy simulation output (`artifacts/growth_uplift/policy_simulation.json`) is a proxy-based targeting exercise under budget constraints, not causal ground truth.
Use `artifacts/growth_uplift/policy_simulation_budget_curve.png` for a quick visual of cumulative proxy lift versus budget spend.

## Docs

- Architecture source of truth: `docs/architecture.pptx`
- Documentation map and conventions: `docs/README.md`

## Dataset

[Base dataset: Online Retail II (UCI transactional dataset)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci/data)

This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.
