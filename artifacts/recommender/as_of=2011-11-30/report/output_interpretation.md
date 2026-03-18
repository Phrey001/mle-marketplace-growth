# Recommender Automated Interpretation

## Model Selection
- Selected model: `mf`
- Selection rule: `maximize_validation_Recall@20`

## Random Baseline Anchor
- Catalog size (N): 3525
- Recommendation depth (K): 20
- Random Recall@20 anchor (K/N): 0.005674

## Validation Snapshot
- Two-tower Recall@20: 0.0376 (lift vs random anchor: 6.62x)
- MF Recall@20: 0.0525 (lift vs random anchor: 9.26x)
- Popularity Recall@20: 0.0283 (lift vs random anchor: 4.98x)

## Test Snapshot
- Two-tower Recall@20: 0.0276 (lift vs random anchor: 4.87x)
- MF Recall@20: 0.0430 (lift vs random anchor: 7.58x)
- Popularity Recall@20: 0.0229 (lift vs random anchor: 4.04x)

_Scope: offline ranking quality only; causal business lift needs online experiment._
