# Purchase Propensity Engine — Spec

High-level architecture lives in `docs/architecture.pptx`.
This file is the implementation contract for the purchase propensity engine.

## Objective

Predict purchase propensity and allocate budget with expected-value targeting:

- `expected_value_score = propensity_score * predicted_conditional_revenue_30d`

## Scope

- In scope: offline predictive policy evaluation.
- Out of scope: causal incrementality claims (requires online experimentation).

## Core Contract

| Area | Contract |
|---|---|
| Split mode | Strict monthly `10/1/1` (10 train + 1 validation + 1 test snapshots) |
| Datetime anchor | `panel_end_date` in cycle YAML; must be a monthly snapshot date (1st of month); derive prior 11 monthly snapshots |
| Structural mode (initial) | `window_selection_mode=sensitivity` |
| Structural mode (retrain) | `window_selection_mode=fixed` |
| Policy evaluation | Budget-constrained Top-K (`K=floor(budget/cost_per_user)`) on validation + test |
| Policy baselines | `ml_top_expected_value`, `random_baseline`, `rfm_heuristic` |
| Scope boundary | Offline evidence only; not causal proof |

Datetime ownership/bounds:
- Shared silver data availability is defined by `configs/shared.yaml`.
- Engine datetime is owned by purchase propensity config (`panel_end_date`).
- Engine datetime may be narrower than shared bounds, but must not exceed shared silver event-date bounds (fail-fast on violation).

## Pipeline Map

| Stage | Script | Key output(s) |
|---|---|---|
| Feature-store build | `mle_marketplace_growth.feature_store.build_gold_purchase_propensity` | 12 snapshot partitions of `propensity_train_dataset`, `user_features_asof` |
| Model training | `mle_marketplace_growth.purchase_propensity.train` | `offline_eval/propensity_model.pkl`, `offline_eval/train_metrics.json`, `offline_eval/validation_predictions.csv`, `offline_eval/test_predictions.csv` |
| Structural sensitivity (initial) | `mle_marketplace_growth.purchase_propensity.window_sensitivity` | `offline_eval/window_sensitivity.json`, `offline_eval/window_validation_dashboard.png` |
| Policy backtest | `mle_marketplace_growth.purchase_propensity.policy_budget_evaluation` | `offline_eval/offline_policy_budget_validation.json`, `offline_eval/offline_policy_budget_test.json` |
| Artifact checks/report text | `mle_marketplace_growth.purchase_propensity.validate_outputs` | `report/output_validation_summary.json`, `report/output_interpretation.md` |
| Serving-style batch scoring | `mle_marketplace_growth.purchase_propensity.predict` | `serving_batch/as_of_date=YYYY-MM-DD/prediction_scores.csv` |

## Model/Window Contract

| Item | Contract |
|---|---|
| Propensity candidates | `logistic_regression`, `xgboost` |
| Propensity selection metric | Validation `average_precision` (PR-AUC) |
| Revenue model | Conditional revenue regressor (XGBoost + fallback baseline) |
| Window sets | `prediction_window_days` in `{30,60,90}`; `feature_lookback_days` in `{60,90,120}` |
| Fixed mode behavior | Uses config values directly; no structural re-search |
| Sensitivity mode behavior | Runs window sensitivity and freezes structural decision |

Feature window notes:
1. SQL: 30d short-term lookback windows (`frequency_30d`, `monetary_30d`) – Always include.
2. SQL: Longer-term feature lookback window days (`frequency_*`, `monetary_*`, `avg_basket_value_*`) – Include depending on lookback window config (e.g. `90d`).
3. Training/serving: Cap longer-term monetary lookback by excluding outliers to reduce skew to fit model towards majority of users.
4. SQL: Skip 30‑day average basket size because most users have too few purchases in 30 days for a stable average. Still keep 30‑day counts and spend totals because even small activity in the last 30d is a useful signal.
5. SQL: `label_*` features – Include depending on prediction window config.

Feature matrix example (`train.py`):
- Before vectorization: one row is a dict, e.g. `{"recency_days": 12.0, "frequency_30d": 0.0, ..., "country": "US"}`.
- After `DictVectorizer.fit/transform`: `train_matrix` is a sparse numeric matrix with shape `(n_train_rows, n_features)`, e.g. `(50000, 8)`.
- Categorical values (like `country`) are encoded into numeric columns by the vectorizer.

## Artifact Contract

Per cycle (`artifacts/purchase_propensity/<cycle>/`):

- `offline_eval/propensity_model.pkl`
- `offline_eval/train_metrics.json`
- `offline_eval/validation_predictions.csv`
- `offline_eval/test_predictions.csv`
- `offline_eval/offline_policy_budget_validation.json`
- `offline_eval/offline_policy_budget_test.json`
- `report/output_validation_summary.json`
- `report/output_interpretation.md`

Initial cycle only (sensitivity mode):

- `offline_eval/window_sensitivity.json`
- `offline_eval/window_validation_dashboard.png`

## Acceptance Criteria

- Output validation summary passes.
- Structural decision is frozen and recorded (or fixed mode is explicit).
- Validation/test policy outputs each contain ML/Random/RFM.
- Serving output is schema-valid when Stage 3 is run.

Run commands: `docs/purchase_propensity/quickstart.md`.
