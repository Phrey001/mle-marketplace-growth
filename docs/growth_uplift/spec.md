# Growth Uplift Engine — Spec Notes (Supplemental)

This document captures implementation/spec details that may be useful for designing components and interfaces.
The high-level architecture and flows live in `docs/architecture.pptx`.

---

## Objective

Allocate promotional treatments (e.g., vouchers/discounts/promotions) to users predicted to generate the highest incremental net revenue, subject to budget/capacity constraints.

---

## Experimental Design (Offline / A/B Simulation)

- Deterministic assignment via hash bucketing
- Fixed post-assignment outcome window (e.g., 30 days)
- Capacity constraint on treated users

Example voucher design (illustrative only):

- 10% discount
- $5 cap

---

## Baseline Modeling (Example)

Baseline revenue can be decomposed into components such as:

- Purchase probability (e.g., via historical Poisson arrival assumption)
- Basket value (e.g., historical median + lognormal noise)

Treatment effects can be applied to:

- Conversion probability
- Basket size

---

## Uplift Modeling (Example)

T-learner:

- Model 1: `E[Net Revenue | Treated, X]`
- Model 2: `E[Net Revenue | Control, X]`
- Individual Treatment Effect (ITE) = difference between the two models

---

## Policy Evaluation

Compare candidate policies such as:

- Treat none
- Treat all
- Random K%
- Top-uplift K%

Metrics (examples):

- Incremental net revenue
- ROI
- Uplift/Qini curve

---

## Deliverables

- User-level uplift scores
- Policy revenue comparison
- Counterfactual/offline policy evaluation report

---

## Current Baseline Implementation

Simulation scope (important):

- Feature-store datasets remain promotion-agnostic and do not contain real campaign treatment logs.
- `train.py` uses synthetic treatment assignment (`treatment` via deterministic hash bucketing) only for offline uplift pipeline validation.
- `evaluation.json` and `evaluation_policy_lift.png` are for model-logic diagnostics, not direct business-impact validation.
- Best-practice interpretation: treat this as a mechanics baseline only; causal business impact requires real treatment logs (or a fully synthetic sandbox where outcomes are also simulated from assumptions).

Training script:

- `src/mle_marketplace_growth/growth_uplift/train.py`
- Input: `data/gold/feature_store/growth_uplift/uplift_train_dataset/as_of_date=<DATE>/uplift_train_dataset.csv`
- Method: baseline T-learner (`treated_model` and `control_model`) with deterministic pseudo-random treatment assignment by hash bucketing.
  - `treated_model` is trained on treated rows only and predicts expected outcome under treatment.
  - `control_model` is trained on control rows only and predicts expected outcome without treatment.
  - Uplift score is computed per user as `pred_treated - pred_control`.
  - Candidate baselines: `random_forest` and `gradient_boosting`
  - Selection rule: pick model with highest `top10_vs_all_users_lift_delta` on validation.
- Outputs:
  - `artifacts/growth_uplift/uplift_model.pkl`
  - `artifacts/growth_uplift/train_metrics.json` (includes `selected_model_name` and candidate comparison)
  - `artifacts/growth_uplift/validation_scores.csv` (includes `treatment`, `observed_outcome`, prediction columns)

Batch scoring script:

- `src/mle_marketplace_growth/growth_uplift/predict.py`
- Input: `user_features_asof` snapshot + trained model bundle
- Output: `artifacts/growth_uplift/prediction_scores.csv` with `uplift_score`.

Evaluation script:

- `src/mle_marketplace_growth/growth_uplift/evaluate.py`
- Input: `artifacts/growth_uplift/validation_scores.csv`
- Outputs:
  - `artifacts/growth_uplift/evaluation.json` (compact policy-comparison table)
  - `artifacts/growth_uplift/evaluation_policy_lift.png` (high-level review plot)

Policy simulation script (budget-constrained targeting):

- `src/mle_marketplace_growth/growth_uplift/policy_simulation.py`
- Inputs:
  - `data/gold/feature_store/growth_uplift/user_features_asof/as_of_date=<DATE>/user_features_asof.csv`
  - `artifacts/growth_uplift/prediction_scores.csv`
- Output:
  - `artifacts/growth_uplift/policy_simulation.json`
  - `artifacts/growth_uplift/policy_simulation_budget_curve.png`
- KPIs:
  - `expected_revenue_lift_proxy_total`
  - `budget_efficiency_lift_per_dollar`

Recommended commands (for meaningful future-window labels):

```bash
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build --as-of-date 2011-11-09
PYTHONPATH=src python -m mle_marketplace_growth.growth_uplift.train --input-csv data/gold/feature_store/growth_uplift/uplift_train_dataset/as_of_date=2011-11-09/uplift_train_dataset.csv
PYTHONPATH=src python -m mle_marketplace_growth.growth_uplift.predict --input-csv data/gold/feature_store/growth_uplift/user_features_asof/as_of_date=2011-11-09/user_features_asof.csv
PYTHONPATH=src python -m mle_marketplace_growth.growth_uplift.evaluate
PYTHONPATH=src python -m mle_marketplace_growth.growth_uplift.policy_simulation
```

Evaluation interpretation guide (high level):

- `evaluation.json` now includes `outcome_summary`, `interpretation_kpis`, and `sanity`.
- Policy lift uses inverse-propensity-style reweighting in plain terms:
  - treated outcomes are scaled by `1 / treatment_rate`
  - control outcomes are scaled by `1 / (1 - treatment_rate)`
  - incremental lift is `reweighted treated - reweighted control`
- Why this scaling exists: treated/control group sizes can differ, so reweighting puts both on a comparable scale before subtraction.
- Start with `sanity.status`:
  - `ok` means no obvious red flags from automated checks.
  - `review` means inspect `sanity.flags` before interpreting uplift quality.
- Key KPI to review: `interpretation_kpis.top10_vs_all_users_lift_delta`.
  - Positive suggests better prioritization in top-scored users.
  - Non-positive suggests weak/no uplift ranking signal.
- Negative lift is still a valid simulation outcome; it means the current model/policy is underperforming, not that the pipeline is broken.
- If flags include `all_outcomes_zero_check_as_of_date_window` or `policy_lift_all_zero`, use an earlier `as_of_date` so the future label window is non-empty.
