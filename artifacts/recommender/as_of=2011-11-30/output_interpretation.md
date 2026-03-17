# Recommender Automated Interpretation

## Model Selection
- Selected model: `mf`
- Selection rule: `maximize_validation_Recall@20`

## Random Baseline Anchor
- Catalog size (N): 3525
- Recommendation depth (K): 20
- Random Recall@20 anchor (K/N): 0.005674

## Validation Snapshot
- Two-tower Recall@20: 0.0376 (lift vs random anchor: 6.63x)
- MF Recall@20: 0.0525 (lift vs random anchor: 9.26x)
- Popularity Recall@20: 0.0263 (lift vs random anchor: 4.64x)

## Test Snapshot
- Two-tower Recall@20: 0.0290 (lift vs random anchor: 5.10x)
- MF Recall@20: 0.0430 (lift vs random anchor: 7.58x)
- Popularity Recall@20: 0.0215 (lift vs random anchor: 3.79x)

_Scope: offline ranking quality only; causal business lift needs online experiment._
