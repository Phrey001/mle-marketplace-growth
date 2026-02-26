# Automated Interpretation

## Model Generalization
- ROC-AUC: 0.7829
- PR-AUC: 0.6964
- Top-decile lift: 2.2938
- Calibration (ECE, lower is better): 0.0440
- Calibration (Brier, lower is better): 0.1807

## Policy Comparison (Budget-Constrained Holdout Outcomes)
- ML expected-value revenue/targeted user: 743.7414
- Random baseline revenue/targeted user: 232.7335
- RFM baseline revenue/targeted user: 744.1167
- ML vs Random delta: 511.0078
- ML vs RFM delta: -0.3754
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
