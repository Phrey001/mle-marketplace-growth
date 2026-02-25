# Automated Interpretation

## Model Generalization
- ROC-AUC: 0.7797
- PR-AUC: 0.5368
- Top-decile lift: 3.0534
- Calibration (ECE, lower is better): 0.0487
- Calibration (Brier, lower is better): 0.1363

## Policy Comparison (Holdout Outcomes)
- ML expected-value revenue/targeted user: 564.2209
- Random baseline revenue/targeted user: 190.4353
- RFM baseline revenue/targeted user: 563.1645
- ML vs Random delta: 373.7856
- ML vs RFM delta: 1.0564

## Budget Policy Summary
- Policy for this section: ml_top_expected_value only (score-based allocation on prediction snapshot).
- This section is planning-oriented and not directly comparable to holdout actual outcomes above.
- Expected value per targeted user: 465.8047
- Expected value per dollar: 93.160948

## Window Validation
- Best prediction window by PR-AUC: 90d (PR-AUC=0.7302).
- Use this for model-signal comparison only; it is not an automatic business-window selection rule.

_Scope note: offline predictive policy evaluation only; not causal promotional incrementality._
