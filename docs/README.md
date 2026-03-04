# Documentation Guide

`docs/architecture.pptx` is the source of truth for the system architecture diagrams and end-to-end flows.

Documentation map:

| Purpose | Primary files |
|---|---|
| Source-of-truth architecture | `docs/architecture.pptx` |
| Domain specs | `docs/purchase_propensity/spec.md`, `configs/purchase_propensity/`, `docs/recommender/spec.md` |
| Runbooks | `docs/purchase_propensity/quickstart.md`, `docs/recommender/quickstart.md` |
| Shared data/feature conventions | `docs/feature_store/overview.md`, `docs/feature_store/dq_spec.md`, `docs/feature_store/data_dictionary.md`, `docs/feature_store/diagrams/` |

## Datetime Selection Strategy

| Scope | Datetime config |
|---|---|
| Shared silver build | `configs/shared.yaml` (`input_csv`, `output_root`) |
| Purchase propensity | `panel_end_date` (single anchor; pipeline derives prior 11 monthly snapshots for strict 10/1/1 split) |
| Recommender | `recommender_min_event_date` / `recommender_max_event_date` |

Contract:
- engine-owned datetime settings may be narrower than shared silver bounds.
- engine-owned datetime settings must not extend beyond shared silver bounds.
- feature-store build fails fast when bounds are violated.
