# Automated Interpretation

## Model Generalization
- ROC-AUC: 0.7075
- PR-AUC: 0.7234
- Top-decile lift: 1.8025
- Calibration (ECE, lower is better): 0.0356
- Calibration (Brier, lower is better): 0.2179

## Policy Comparison (Budget-Constrained Holdout Outcomes)
- ML expected-value revenue/targeted user: 1314.6944
- Random baseline revenue/targeted user: 509.8546
- RFM baseline revenue/targeted user: 1321.1936
- ML vs Random delta: 804.8397
- ML vs RFM delta: -6.4992
- Validation policy rows: 3
- Test policy rows: 3

## Budget Policy Summary
- Policy comparison uses equal budget-constrained Top-K across ML expected value, random baseline, and RFM baseline.
- This section reports offline holdout outcomes only; no causal incrementality claim.
- ML targeted users (validation): 1000
- ML targeted users (test): 1000

## Window Validation
- Best prediction window by PR-AUC: 90d (PR-AUC=0.7101).
- Use this for model-signal comparison only; it is not an automatic business-window selection rule.

_Scope note: offline predictive policy evaluation only; not causal promotional incrementality._
