# Documentation Guide

`docs/architecture.pptx` is the source of truth for the system architecture diagrams and end-to-end flows.

The Markdown documents are organized by purpose:

- **Source-of-truth architecture**
  - `docs/architecture.pptx`
- **Domain spec notes**
  - `docs/purchase_propensity/spec.md`
  - `configs/purchase_propensity/`
  - `docs/recommender/spec.md`
  - (Each engine spec includes its own tech stack section)
- **Runbook**
  - `docs/purchase_propensity/quickstart.md`
  - `docs/recommender/quickstart.md`
- **Shared data/feature conventions**
  - `docs/feature_store/overview.md`
  - `docs/feature_store/dq_spec.md`
  - `docs/feature_store/data_dictionary.md`
  - `docs/feature_store/diagrams/` (Mermaid source and rendered diagram assets)

## Datetime Selection Strategy

- `configs/shared.yaml` defines shared build paths (`input_csv`, `output_root`) for the shared silver build.
- Engine configs define engine-specific time windows/snapshots:
  - purchase propensity: `train_*` dates and `score_as_of_date`
  - recommender: `recommender_min_event_date` / `recommender_max_event_date`
- Contract:
  - engine-owned datetime settings may be narrower than shared silver bounds
  - engine-owned datetime settings must not extend beyond shared silver bounds
  - feature-store build fails fast when bounds are violated
