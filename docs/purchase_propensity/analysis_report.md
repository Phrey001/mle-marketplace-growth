# Purchase Propensity — Analytical Report

_Based on latest local artifacts in `artifacts/purchase_propensity/cycle_initial/` and `artifacts/purchase_propensity/cycle_retrain/`._

_Update this report or append new version (not automated) after each recommended quickstart run if there are significant updates._

## 1) Executive Summary

**Operational call to action:** deploy ML expected-value targeting as the default candidate for budget-constrained allocation, keep RFM as the live challenger, and treat results as offline evidence (not causal proof).

| Area | Decision | Executive Intuition |
|---|---|---|
| Run health | Proceed | Both cycles passed automated validation checks, so outputs are internally consistent. |
| Policy | Use ML expected-value targeting as default candidate | ML materially outperforms random targeting in revenue per contacted user under the same budget. |
| Benchmark | Keep RFM as active benchmark | RFM is still slightly stronger on revenue per targeted user, so benchmark pressure should remain. |
| Why ML over RFM (for default candidate) | Keep ML as operating default while benchmarking against RFM each cycle | RFM is a strong baseline today, but ML is a better long-term targeting engine: it already beats random by a wide margin, ranks by expected value (`purchase_probability × predicted_revenue`), scales to richer feature space, and supports iterative model tuning while RFM remains the live benchmark guardrail. |
| Structural setup | Freeze `90d` prediction + `120d` lookback + `xgboost` from initial cycle; keep fixed for retrain | Stable structure reduces process complexity and improves comparability across retraining cycles. |
| Scope | Offline policy evaluation only | Results are decision-support signals, not causal incrementality proof. |

## 2) Evaluation Setup

**YAML-configurable knobs** for this run (`configs/purchase_propensity/demo_cycle_initial.yaml` and `configs/purchase_propensity/demo_cycle_retrain.yaml`):
- Validation mode: `out_of_time_10_1_1` (strict 10 train months + 1 validation month + 1 test month)
- Policy comparison mode: budget-constrained `Top-K` per slice (`K = budget / cost_per_user`)
- Budget policy assumptions: budget `5000`, cost/user `5`
- Window knobs: `prediction_window_days`, `feature_lookback_days` (defined in YAML)

**Validation slicing note:** this run uses strict monthly `10/1/1` slices per cycle (`train/validation/test`).
Method details are defined in `docs/purchase_propensity/spec.md`.

**Observed run outputs** (artifact-derived, informational):
- `cycle_initial/` and `cycle_retrain/` each include strict split predictions (`validation_predictions.csv`, `test_predictions.csv`), budget policy outputs, and output validation/interpretation files.
- Serving snapshot scoring writes `prediction_scores.csv` per cycle for operational targeting handoff; policy conclusions below are based on holdout outcomes.

## 3) Model Quality (Validation Slice)

**Objective:** verify the model has usable ranking and calibration quality before policy comparison.

| Cycle | Selected model | ROC-AUC | PR-AUC | Top-decile lift | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| Initial batch (first-year panel) | xgboost | 0.707477 | 0.723371 | 1.802465 | 0.217854 | 0.035603 |
| Single rolling retrain (Q4 2011) | xgboost | 0.782887 | 0.696431 | 2.293817 | 0.180747 | 0.044048 |

Revenue-model validation quality (buyers only):

| Cycle | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Initial batch (first-year panel) | 2383.351261 | 717.361837 | 1.303019 |
| Single rolling retrain (Q4 2011) | 4408.665581 | 708.839116 | 1.480579 |

## 4) Policy Backtest Results (Holdout Outcomes)

**Objective:** compare targeting policies on realized holdout outcomes using the same target volume.

**Interpretation:**
- ML targeting clearly beats random baseline in both cycles.
- ML remains slightly below RFM in both cycles on revenue per targeted user.

### Cycle 1: Initial Batch (first-year panel)

| Policy | High-level policy rule | Revenue / targeted user | Purchase rate |
|---|---|---:|---:|
| ML expected value | Rank by `propensity_score × predicted_conditional_revenue`, target Top-K by budget | 1314.694361 | 0.6660 |
| Random baseline | Deterministic random target selection, target Top-K by budget | 509.854640 | 0.4110 |
| RFM heuristic | Rank by recency/frequency/monetary heuristic, target Top-K by budget | 1321.193591 | 0.6710 |

- ML vs Random revenue/targeted-user delta: `+804.839721`
- ML vs RFM revenue/targeted-user delta: `-6.499230`

### Cycle 2: Single Rolling Retrain (Q4 2011)

| Policy | High-level policy rule | Revenue / targeted user | Purchase rate |
|---|---|---:|---:|
| ML expected value | Rank by `propensity_score × predicted_conditional_revenue`, target Top-K by budget | 743.741380 | 0.5860 |
| Random baseline | Deterministic random target selection, target Top-K by budget | 232.733540 | 0.2870 |
| RFM heuristic | Rank by recency/frequency/monetary heuristic, target Top-K by budget | 744.116740 | 0.5850 |

- ML vs Random revenue/targeted-user delta: `+511.007840`
- ML vs RFM revenue/targeted-user delta: `-0.375360`

Policy comparison details are stored in:
- `artifacts/purchase_propensity/cycle_initial/offline_policy_budget_test.json`
- `artifacts/purchase_propensity/cycle_retrain/offline_policy_budget_test.json`

![Policy Comparison by Cycle](../../artifacts/purchase_propensity/report_assets/policy_comparison_cycles.png)

## 5) Window Sensitivity + Freeze Decision

Initial cycle freeze decision:
- selected prediction window: `90d`
- selected feature lookback: `120d`
- selected propensity model: `xgboost`

| Cycle | Best prediction window by PR-AUC | PR-AUC |
|---|---|---:|
| Initial batch (first-year panel) | 90d | 0.7101 |

Window outputs:
- `artifacts/purchase_propensity/cycle_initial/window_sensitivity.json`

Retrain cycle uses fixed structural settings from the initial freeze (`window_selection_mode=fixed`), so no structural re-search is required.

![Initial Cycle Window Validation Dashboard](../../artifacts/purchase_propensity/cycle_initial/window_validation_dashboard.png)

## 6) Two-Cycle Comparison Summary

| Cycle | Score Date | Model | ROC-AUC | PR-AUC | RMSE | MAE | MAPE | ML vs Random (rev/user) | ML vs RFM (rev/user) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Initial batch (first-year panel) | 2010-11-09 | xgboost | 0.707477 | 0.723371 | 2383.351261 | 717.361837 | 1.303019 | 804.839721 | -6.499230 |
| Single rolling retrain (Q4 2011) | 2011-11-09 | xgboost | 0.782887 | 0.696431 | 4408.665581 | 708.839116 | 1.480579 | 511.007840 | -0.375360 |

Interpretation notes:
- Ranking discrimination (`ROC-AUC`) and concentration (`Top-decile lift`) are stronger in retrain cycle.
- PR-AUC is slightly higher in the initial cycle; keep both metrics in interpretation.
- ML policy beats random baseline in both cycles by large margin.
- ML remains slightly below RFM in both cycles; the gap narrows materially in retrain cycle.
- Revenue-regression error increases in retrain cycle; evaluate alongside policy outcomes when discussing tradeoffs.
