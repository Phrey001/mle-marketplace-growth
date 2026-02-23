# Growth Uplift Engine (Growth Decisioning)

Who to treat, and how much incremental uplift to expect, under budget/capacity constraints (e.g., vouchers, discounts, promotions).

This document is aligned to `docs/architecture.pptx`.

---

## Objective

Maximize incremental revenue by allocating a limited treatment budget to users with the highest expected uplift.

---

## Engine Flow

- **Preprocessing in Feature Store:** RFM-style behavioural aggregation at user level
- **Training Pipeline (Offline):** baseline + treatment effect modelling
- **Serving (Policy) Pipeline:** policy optimization (e.g., select top uplift users subject to budget)
- **Offline Policy Evaluation (A/B Simulation):**
  - Simulate baseline policy vs uplift policy
  - Compare total revenue, incremental lift, and revenue per user
- **Periodic Retraining:** if running in a production environment

---

## Uplift Model (Definition)

An uplift model answers: “How much extra revenue will this user generate if treated vs not treated?”

One common framing:

- **Baseline model:** estimate revenue if not treated
- **Treatment model:** estimate revenue if treated
- **Uplift:** treatment − baseline (expected incremental revenue)

---

## Outputs

- User-level uplift scores
- Budget-/capacity-constrained treatment policy
- Offline policy evaluation report (baseline vs uplift policy)

See also: `docs/growth_uplift_spec.md` (supplemental implementation/spec notes).
