# Automated Interpretation

## Model Generalization
- ROC-AUC: 0.7100
- PR-AUC: 0.7467
- Top-decile lift: 1.7492
- Calibration (ECE, lower is better): 0.0329
- Calibration (Brier, lower is better): 0.2161

## Policy Comparison (Budget-Constrained Holdout Outcomes)
- ML expected-value revenue/targeted user: 1432.0304
- Random baseline revenue/targeted user: 505.7851
- RFM baseline revenue/targeted user: 1444.3692
- ML vs Random delta: 926.2454
- ML vs RFM delta: -12.3388
- Validation policy rows: 3
- Test policy rows: 3

## Budget Policy Summary
- Policy comparison uses equal budget-constrained Top-K across ML expected value, random baseline, and RFM baseline.
- This section reports offline holdout outcomes only; no causal incrementality claim.
- ML targeted users (validation): 1000
- ML targeted users (test): 1000

## Window Validation
- Best prediction window by PR-AUC: 90d (PR-AUC=0.7313).
- Use this for model-signal comparison only; it is not an automatic business-window selection rule.

_Scope note: offline predictive policy evaluation only; not causal promotional incrementality._
