# Purchase Propensity Engine — Spec

High-level architecture lives in `docs/architecture.pptx`.
This doc is the implementation-facing spec for the purchase propensity engine.
Temporal snapshot schedule is documented in `docs/purchase_propensity/snapshot_plan.md`.

## Objective

Predict which users are most likely to purchase in the next 30 days, then allocate incentives under a budget constraint by targeting the highest expected-value users.

Expected value in this repo:

- `expected_value_score = propensity_score * predicted_conditional_revenue_30d`

## Scope and Causality

- This project evaluates incentive allocation strategies under a predictive expected-value framework.
- This repository performs **offline policy backtesting**, not causal promotional incrementality estimation.
- Policy comparison uses holdout-window realized outcomes (`label_purchase_30d`, `label_net_revenue_30d`).
- True incremental business impact still requires randomized online experimentation.

## Current Pipeline

### 1) Feature Store Build

| Output | Purpose |
|---|---|
| `data/gold/feature_store/purchase_propensity/propensity_train_dataset/...` | Training/evaluation dataset with labels |
| `data/gold/feature_store/purchase_propensity/user_features_asof/...` | Batch scoring snapshot |

### 2) Train Propensity Models

| Item | Design |
|---|---|
| Script | `src/mle_marketplace_growth/purchase_propensity/train.py` |
| Input | `propensity_train_dataset.csv` |
| Label | `label_purchase_30d` |
| Candidate models | `logistic_regression` (baseline), `xgboost` (primary) |
| Selection KPI | Highest validation `average_precision` |
| Scaling | StandardScaler for logistic regression path |
| Spend capping | Cap `monetary_90d` at quantile (default `q=0.99`) before fit/scoring |
| Calibration | Platt scaling (`sigmoid`) by default |

Business-friendly rationale:
- Spend capping stabilizes feature inputs so extreme spend histories do not over-distort propensity/revenue model fit.
- Calibration improves trust in score meaning during planning (for example, “0.7 score” aligns better with observed purchase likelihood).

Split mode used by `run_pipeline.py`:

| Mode | Meaning | Recommended use |
|---|---|---|
| `out_of_time_10_1_1` (implicit) | Requires exactly 12 snapshots and splits into 10 train, 1 validation, 1 test month | Strict architecture-aligned demo cycle |

Out-of-time slice selection:
- Snapshot panel is monthly (`train_frequency=monthly`).
- Train snapshots come from `train_as_of_dates` or generated `train_start_date..train_end_date`.
- `out_of_time_10_1_1`: strict split = first 10 monthly slices train, 11th slice validation, 12th slice test.

Training outputs:

| Artifact | Purpose |
|---|---|
| `artifacts/purchase_propensity/propensity_model.pkl` | Trained model bundle |
| `artifacts/purchase_propensity/train_metrics.json` | Model metrics + out-of-time quality KPIs |
| `artifacts/purchase_propensity/validation_predictions.csv` | Validation-slice predictions with ML/random/RFM policy scores |
| `artifacts/purchase_propensity/test_predictions.csv` | Test-slice predictions with ML/random/RFM policy scores |

Out-of-time quality KPIs captured in `train_metrics.json`:
- ROC-AUC
- PR-AUC (`average_precision`)
- Top-decile lift
- Calibration metrics (Brier score, ECE with 10 bins)
- Revenue-model holdout quality (on buyers only): RMSE, MAE, MAPE

### 3) Offline Policy Backtest (Budget-Constrained Holdout)

Budget-constrained policy backtest compares policies on holdout slices (validation/test) with equal `Top-K` count derived from `budget / cost_per_user`:

| Policy | Selection rule |
|---|---|
| `ml_top_expected_value` | Top-K by expected value |
| `random_baseline` | Top-K by deterministic random score |
| `rfm_heuristic` | Top-K by heuristic RFM score |

Backtest metrics:

- actual purchase rate of targeted segment
- actual revenue total of targeted segment
- actual revenue per targeted user

Evaluation artifacts:

| Artifact | Purpose |
|---|---|
| `artifacts/purchase_propensity/offline_policy_budget_validation.json` | Budget-constrained policy metrics on validation slice |
| `artifacts/purchase_propensity/offline_policy_budget_test.json` | Budget-constrained policy metrics on test slice |
| `artifacts/purchase_propensity/output_validation_summary.json` | Automated output sanity-check summary |
| `artifacts/purchase_propensity/output_interpretation.md` | Automated interpretation summary |

### 4) Budgeted Offline Policy Evaluation

| Item | Design |
|---|---|
| Script | `src/mle_marketplace_growth/purchase_propensity/policy_budget_evaluation.py` |
| Input | `validation_predictions.csv` / `test_predictions.csv` |
| Allocation rule | Equal `Top-K` by budget per policy (`K = floor(budget / cost_per_user)`) |
| Outputs | `offline_policy_budget_validation.json`, `offline_policy_budget_test.json` |

Budget-constrained policy comparison on holdout slices (validation/test) is also produced with equal budget-based target count (`Top-K by budget`) for:
- `ml_top_expected_value`
- `random_baseline`
- `rfm_heuristic`

Executable commands are documented in `docs/quickstart.md`.

### 5) Serving Snapshot Scoring (Operational Output)

| Item | Design |
|---|---|
| Script | `src/mle_marketplace_growth/purchase_propensity/predict.py` |
| Input | `user_features_asof.csv` + `propensity_model.pkl` |
| Output | `artifacts/purchase_propensity/prediction_scores.csv` |
| Purpose | Operational ranked list for user-level targeting handoff (not a holdout evaluation artifact) |

## Split Semantics (Train / Val / Test in this repo)

- `train.py` uses train rows to fit candidate models.
- Candidate selection is based on holdout ranking quality (`average_precision`) from out-of-time validation slices.
- Final model/policy evaluation and interpretation artifacts are computed on the strict test slice (12th snapshot), with separate validation-slice policy backtest artifacts also emitted.
- In the strict run-pipeline mode, split ratios are fixed (10/1/1) and do not use `validation_rate`.

## Window Choice Note (30 days)

The main training target remains `label_purchase_30d`.
Feature-store gold tables now also materialize 60/90-day labels and 60/120-day lookback features for sensitivity analysis.

Cadence and window alignment in current pipeline:
- Generated snapshot panels use monthly cadence (`train_frequency=monthly`).
- `prediction_window_days` and `feature_lookback_days` are separate config knobs.
- Allowed sets are explicit: `prediction_window_days` `{30,60,90}` and `feature_lookback_days` `{60,90,120}`.
- Training panel period is config-driven (`train_start_date`, `train_end_date` in cycle YAML configs).
- Strict split is fixed to 10 train months + 1 validation month + 1 test month per cycle.
- `run_pipeline.py` supports `window_selection_mode`:
  - `sensitivity`: run `window_sensitivity.py` and freeze by validation PR-AUC.
  - `fixed`: use config-provided structural choices directly (no reopen of structural search).

`window_sensitivity.py` runs model metrics for 30/60/90 prediction windows and 60/90/120 lookback profiles, and writes:

- `artifacts/purchase_propensity/window_sensitivity.json`

The same output also includes:
- inter-purchase gap distribution summary
- prediction-window validation against inter-purchase gap coverage
- feature-lookback validation (60/90/120 profiles) using model metrics
- `window_validation_dashboard.png` for visual comparison of PR-AUC, top-decile lift, Brier, and ECE

How to read window-sensitivity metrics (concise):

| Metric | Read as |
|---|---|
| ROC-AUC | Overall ranking quality (higher is better) |
| PR-AUC | Primary positive-class retrieval quality for model selection |
| Top-decile lift | Buyer concentration in top-ranked users |
| Brier | Probability error (lower is better) |
| ECE | Calibration gap vs observed rates (lower is better) |
| Inter-purchase gap coverage | Share of observed repeat-purchase gaps captured by window |

Interpretation guideline:
- Use inter-purchase gap coverage to check business horizon fit.
- Use PR-AUC as primary model-signal comparator across windows.
- Use ROC-AUC / Top-decile lift as ranking sanity checks.
- Use Brier / ECE to watch calibration-quality tradeoffs.

Sensitivity runs on the merged strict 12-snapshot panel and follows the same chronological 10/1/1 split semantics (train/validation/test dates).

`train.py` model validation now includes both:
- propensity candidate validation (logistic regression vs xgboost on validation PR-AUC/ROC-AUC)
- revenue candidate validation (xgboost regressor vs constant baseline on validation RMSE/MAE/MAPE over buyer rows)

Policy backtest outputs are produced for both validation and test slices:
- `offline_policy_budget_validation.json`
- `offline_policy_budget_test.json`

## Retraining Cadence

- Current repo orchestration is snapshot-driven and manual/CLI-triggered.
- Demo run pattern: initial 12-month cycle + one rolling retrain cycle, both with strict 10/1/1 chronology.
- Recommended practice in this repo: freeze structural decisions on the initial cycle (`window_selection_mode=sensitivity`), then keep retrain cycle fixed (`window_selection_mode=fixed`).

## Testing

Integration test that mirrors the recommended run flow (using a compact fixture dataset):
- `tests/test_purchase_propensity_integration.py`
- Test execution commands are documented in `docs/quickstart.md`.
