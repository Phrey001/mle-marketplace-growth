# Recommender Automated Interpretation

## Model Selection
- Selected model: `mf`
- Selection rule: `maximize_validation_Recall@20`

## Random Baseline Anchor
- Catalog size (N): 3644
- Recommendation depth (K): 20
- Random Recall@20 anchor (K/N): 0.005488

## Validation Snapshot
- Two-tower Recall@20: 0.0996 (lift vs random anchor: 18.14x)
- MF Recall@20: 0.1250 (lift vs random anchor: 22.77x)
- Popularity Recall@20: 0.0564 (lift vs random anchor: 10.27x)

## Test Snapshot
- Two-tower Recall@20: 0.0727 (lift vs random anchor: 13.25x)
- MF Recall@20: 0.0955 (lift vs random anchor: 17.40x)
- Popularity Recall@20: 0.0672 (lift vs random anchor: 12.24x)

_Scope: offline ranking quality only; causal business lift needs online experiment._
