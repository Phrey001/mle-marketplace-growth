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

Validation modes:

| Mode | Meaning | Recommended use |
|---|---|---|
| `hash` | Deterministic user/date split | Lightweight debugging and sanity checks |
| `out_of_time` | Hold out latest `as_of_date` buckets (requires multiple snapshots) | Preferred for production-like model validation |

Out-of-time slice selection (current behavior):
- Snapshot panel is monthly (`train_frequency=monthly`).
- Train snapshots come from `train_as_of_dates` or generated `train_start_date..train_end_date`.
- `train.py` sorts unique `as_of_date` values and holds out the latest fraction by `validation_rate`.
- Holdout slice count formula: `max(1, round(num_snapshots * validation_rate))`.
- Validation = latest monthly slices; training = earlier slices.

Training outputs:

| Artifact | Purpose |
|---|---|
| `artifacts/purchase_propensity/propensity_model.pkl` | Trained model bundle |
| `artifacts/purchase_propensity/train_metrics.json` | Model metrics + policy comparison + out-of-time quality KPIs |
| `artifacts/purchase_propensity/validation_predictions.csv` | Holdout predictions and policy flags |

Out-of-time quality KPIs captured in `train_metrics.json`:
- ROC-AUC
- PR-AUC (`average_precision`)
- Top-decile lift
- Calibration metrics (Brier score, ECE with 10 bins)

### 3) Offline Policy Backtest

Policies compared on validation holdout:

| Policy | Selection rule |
|---|---|
| `ml_top_expected_value` | Top 20% by expected value |
| `random_baseline` | Deterministic random 20% |
| `rfm_heuristic` | Top 20% by heuristic RFM score |

Backtest metrics:

- actual purchase rate of targeted segment
- actual revenue total of targeted segment
- actual revenue per targeted user

Evaluation artifacts:

| Artifact | Purpose |
|---|---|
| `artifacts/purchase_propensity/evaluation.json` | Holdout policy metrics |
| `artifacts/purchase_propensity/evaluation_policy_comparison.png` | Policy comparison visualization |
| `artifacts/purchase_propensity/evaluation_model_diagnostics.png` | Model diagnostics (ROC, PR, calibration, lift) |
| `artifacts/purchase_propensity/output_validation_summary.json` | Automated output sanity-check summary |
| `artifacts/purchase_propensity/output_interpretation.md` | Automated interpretation summary |

### 4) Budgeted Offline Policy Evaluation

| Item | Design |
|---|---|
| Script | `src/mle_marketplace_growth/purchase_propensity/offline_policy_evaluation.py` |
| Input | `artifacts/purchase_propensity/prediction_scores.csv` |
| Allocation rule | Target highest `expected_value_score` until budget is exhausted |
| Outputs | `offline_policy_evaluation.json` (includes expected value per targeted user and per dollar), `offline_policy_evaluation_budget_curve.png` |

Executable commands are documented in `docs/quickstart.md`.

## Window Choice Note (30 days)

The main training target remains `label_purchase_30d`.
Feature-store gold tables now also materialize 60/90-day labels and 60/120-day lookback features for sensitivity analysis.

Cadence and window alignment in current pipeline:
- Generated snapshot panels use monthly cadence (`train_frequency=monthly`).
- `prediction_window_days` and `feature_lookback_days` are separate config knobs.
- Allowed sets are explicit: `prediction_window_days` `{30,60,90}` and `feature_lookback_days` `{60,90,120}`.
- Training panel period is config-driven (`train_start_date`, `train_end_date` in `configs/purchase_propensity/default_out_of_time.yaml`).
- Validation split fraction is config-driven (`validation_rate` in `configs/purchase_propensity/default_out_of_time.yaml`).
- Main pipeline execution is still wired to 30-day labels and 90-day feature lookback in `run_pipeline.py`; this is an intentional guardrail while sensitivity evidence is being consolidated.

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

Important: run sensitivity with an earlier `as_of_date` (for example `2011-09-09`). If `as_of_date` is close to the dataset end, longer windows (60/90) are right-censored and may collapse to the same outcomes as 30 days.

## Testing

Integration test that mirrors the recommended run flow (using a compact fixture dataset):
- `tests/test_purchase_propensity_integration.py`
- Test execution commands are documented in `docs/quickstart.md`.
