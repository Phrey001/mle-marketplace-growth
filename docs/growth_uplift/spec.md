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

