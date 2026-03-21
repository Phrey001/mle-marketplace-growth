# Purchase Propensity — Spec

High-level architecture lives in `docs/architecture.pptx`.
This file is the design contract for the purchase propensity system.

## What This System Does

- Estimates which users are most likely to purchase in the near term
- Predicts conditional revenue so users can be ranked by expected value, not just raw propensity
- Compares ML targeting against Random and RFM policy baselines under a fixed budget
- Separates structural window selection from later fixed-window retraining

## Objective

Predict purchase propensity and allocate budget with expected-value targeting:

- `expected_value_score = propensity_score * predicted_conditional_revenue_30d`

## Scope

- In scope: offline predictive policy evaluation.
- Out of scope: causal incrementality claims (requires online experimentation).

## Core Contract

Non-negotiable behavior rules:

| Area | Contract |
|---|---|
| Split mode | Strict monthly `10/1/1` (10 train + 1 validation + 1 test snapshots) |
| Datetime anchor | Each run is anchored to a monthly snapshot date and uses the prior 11 monthly snapshots |
| Structural mode (initial) | `window_selection_mode=sensitivity` |
| Structural mode (retrain) | `window_selection_mode=fixed` |
| Policy evaluation | Budget-constrained Top-K (`K=floor(budget/cost_per_user)`) on validation + test |
| Policy baselines | `ml_top_expected_value`, `random_baseline`, `rfm_heuristic` |
| Scope boundary | Offline evidence only; not causal proof |

Datetime ownership/bounds:
- Shared source data defines the maximum available event-date range.
- Each purchase propensity run chooses one monthly snapshot date within that shared range.
- The system may operate on a narrower window than the shared data availability, but not beyond it.

## System Flow

High-level lifecycle only:

1. Build monthly user snapshots and point-in-time features.
2. Train the propensity and conditional revenue models.
3. Run structural window sensitivity in the initial cycle, then freeze those decisions for retraining.
4. Backtest budget-constrained targeting policies on validation and test.
5. Validate the outputs and generate interpretation text.
6. Optionally materialize serving-style batch scores.

## Model/Window Contract

Modeling and window-selection rules:

| Item | Contract |
|---|---|
| Propensity candidates | `logistic_regression`, `xgboost` |
| Propensity selection metric | Validation `average_precision` (PR-AUC) |
| Revenue model | Conditional revenue regressor (XGBoost + fallback baseline) |
| Window sets | `prediction_window_days` in `{30,60,90}`; `feature_lookback_days` in `{60,90,120}` |
| Fixed mode behavior | Uses config values directly; no structural re-search |
| Sensitivity mode behavior | Runs window sensitivity and freezes structural decision |

Feature window rules:

| Feature group | Rule |
|---|---|
| 30d short-term lookback (`frequency_30d`, `monetary_30d`) | Always include |
| Longer-term lookback (`frequency_*`, `monetary_*`, `avg_basket_value_*`) | Include according to configured lookback window |
| Longer-term monetary features | Cap outliers during training/serving to reduce skew toward extreme users |
| 30d average basket size | Omit because most users have too few purchases in 30 days for a stable average |
| 30d counts and spend totals | Keep because even small recent activity is still useful signal |
| `label_*` features | Include according to configured prediction window |

## Output Contract

Output categories only; exact commands live in the purchase propensity quickstart.

Per cycle outputs:
- trained model state
- training summary
- validation and test predictions
- validation and test policy backtest summaries
- validation summary
- interpretation markdown

Initial sensitivity cycle only:
- window sensitivity summary
- sensitivity dashboard

## Acceptance Criteria

- Output validation summary passes.
- Structural decision is frozen and recorded (or fixed mode is explicit).
- Validation/test policy outputs each contain ML/Random/RFM.
- Serving output is schema-valid when Stage 3 is run.
