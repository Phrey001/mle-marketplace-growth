# Purchase Propensity Engine — Spec

High-level architecture lives in `docs/architecture.pptx`.
This doc is the implementation-facing spec for the purchase propensity engine.
Temporal snapshot schedule is defined directly in cycle configs under `configs/purchase_propensity/`.

## Spec Lifecycle

- This spec starts as a preliminary design document derived from `docs/architecture.pptx`.
- As code and run artifacts stabilize, it is finalized as the implementation contract for this repo.

## Objective

Predict which users are most likely to purchase in the next 30 days, then allocate incentives under a budget constraint by targeting the highest expected-value users.

Expected value in this repo:

- `expected_value_score = propensity_score * predicted_conditional_revenue_30d`

## Scope and Causality

- This project evaluates incentive allocation strategies under a predictive expected-value framework.
- This repository performs **offline policy backtesting**, not causal promotional incrementality estimation.
- Policy comparison uses holdout-window realized outcomes (`label_purchase_30d`, `label_net_revenue_30d`).
- True incremental business impact still requires randomized online experimentation.

## Tech Stack

- Feature/data layer: DuckDB SQL + CSV materialization.
- Propensity modeling: scikit-learn (`logistic_regression`) and XGBoost (`xgboost`).
- Revenue modeling: XGBoost regressor with constant fallback baseline for comparison.
- Backtest/evaluation: Python + JSON/CSV artifacts + matplotlib report assets.

## At-a-Glance Contract

| Area | Contract |
|---|---|
| Split mode | Strict monthly `10/1/1` (10 train + 1 validation + 1 test snapshots) |
| Datetime anchor | `panel_end_date` in cycle YAML; pipeline derives prior 11 monthly snapshots |
| Structural selection | Initial cycle: `window_selection_mode=sensitivity` (freeze by validation PR-AUC) |
| Retrain mode | Retrain cycle: `window_selection_mode=fixed` (no structural re-search) |
| Policy evaluation | Budget-constrained Top-K (`K=floor(budget/cost_per_user)`) on validation and test slices |
| Main policy baselines | `ml_top_expected_value`, `random_baseline`, `rfm_heuristic` |
| Scope boundary | Offline backtest evidence only; not causal incrementality proof |

## Pipeline Map

| Stage | Script | Key output(s) |
|---|---|---|
| Feature-store build | `mle_marketplace_growth.feature_store.build` | `propensity_train_dataset`, `user_features_asof` |
| Model training | `mle_marketplace_growth.purchase_propensity.train` | `propensity_model.pkl`, `train_metrics.json`, `validation_predictions.csv`, `test_predictions.csv` |
| Structural sensitivity (initial cycle) | `mle_marketplace_growth.purchase_propensity.window_sensitivity` | `window_sensitivity.json`, `window_validation_dashboard.png` |
| Policy backtest | `mle_marketplace_growth.purchase_propensity.policy_budget_evaluation` | `offline_policy_budget_validation.json`, `offline_policy_budget_test.json` |
| Serving snapshot scoring | `mle_marketplace_growth.purchase_propensity.predict` | `prediction_scores.csv` |

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
- Snapshot panel is monthly.
- Train snapshots are derived from `panel_end_date` (same day-of-month anchor): previous 11 monthly snapshots + panel end snapshot.
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

Execution + artifacts:

| Item | Design |
|---|---|
| Policy-eval script | `src/mle_marketplace_growth/purchase_propensity/policy_budget_evaluation.py` |
| Inputs | `validation_predictions.csv` / `test_predictions.csv` |
| Allocation rule | Equal `Top-K` by budget (`K = floor(budget / cost_per_user)`) |
| Policy outputs | `offline_policy_budget_validation.json`, `offline_policy_budget_test.json` |
| Validation/report outputs | `output_validation_summary.json`, `output_interpretation.md` |

Executable commands are documented in `docs/purchase_propensity/quickstart.md`.

### 4) Serving Snapshot Scoring (Operational Output)

| Item | Design |
|---|---|
| Script | `src/mle_marketplace_growth/purchase_propensity/predict.py` |
| Input | `user_features_asof.csv` + `propensity_model.pkl` |
| Output | `artifacts/purchase_propensity/prediction_scores.csv` |
| Purpose | Operational ranked list for user-level targeting handoff (not a holdout evaluation artifact) |

## Split and Window Contract

Core semantics:
- `train.py` fits on train rows; model quality selection uses validation slices.
- Final policy evidence is reported on strict test slices (with validation-slice policy outputs also emitted).
- Strict mode is fixed `10/1/1` and does not use `validation_rate`.

Window and cadence:
- Main target remains `label_purchase_30d`.
- Gold tables also materialize 60/90-day labels and 60/120-day lookback features for sensitivity analysis.
- Allowed sets: `prediction_window_days` `{30,60,90}` and `feature_lookback_days` `{60,90,120}`.
- `run_pipeline.py` supports:
  - `sensitivity`: run `window_sensitivity.py` and freeze by validation PR-AUC.
  - `fixed`: use config-provided structural choices directly (no reopen of structural search).

Window sensitivity outputs:
- `artifacts/purchase_propensity/window_sensitivity.json`
- `window_validation_dashboard.png` (PR-AUC, top-decile lift, Brier, ECE visual summary)
- includes inter-purchase gap distribution and prediction/lookback validation summaries

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

Sensitivity runs on the merged strict 12-snapshot panel and follows the same chronological `10/1/1` split semantics.

## Retraining Cadence

- Current repo orchestration is snapshot-driven and manual/CLI-triggered.
- Target design is quarterly rolling retraining on overlapping 12-month windows (strict 10/1/1 within each cycle).
- Demo run pattern in this repo: initial 12-month cycle + one rolling retrain cycle only, to keep scope concise.
- Recommended practice in this repo: freeze structural decisions on the initial cycle (`window_selection_mode=sensitivity`), then keep retrain cycle fixed (`window_selection_mode=fixed`).

## Testing

Integration test that mirrors the recommended run flow (using a compact fixture dataset):
- `tests/test_purchase_propensity_integration.py`
- Test execution commands are documented in `docs/purchase_propensity/quickstart.md`.
