# Automated Interpretation

## Model Generalization
- ROC-AUC: 0.7673
- PR-AUC: 0.6583
- Top-decile lift: 2.5158
- Calibration (ECE, lower is better): 0.1521
- Calibration (Brier, lower is better): 0.1956

## Policy Comparison (Budget-Constrained Holdout Outcomes)
- ML expected-value revenue/targeted user: 956.0462
- Random baseline revenue/targeted user: 323.1149
- RFM baseline revenue/targeted user: 949.3128
- ML vs Random delta: 632.9313
- ML vs RFM delta: 6.7334
- Validation policy rows: 3
- Test policy rows: 3

## Budget Policy Summary
- Policy comparison uses equal budget-constrained Top-K across ML expected value, random baseline, and RFM baseline.
- This section reports offline holdout outcomes only; no causal incrementality claim.
- ML targeted users (validation): 1000
- ML targeted users (test): 1000

## Window Validation
- Window sensitivity not run in fixed mode.
- Use this for model-signal comparison only; it is not an automatic business-window selection rule.

_Scope note: offline predictive policy evaluation only; not causal promotional incrementality._
