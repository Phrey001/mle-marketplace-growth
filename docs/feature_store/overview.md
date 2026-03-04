# Feature Store Overview

Purpose: produce reproducible point-in-time datasets for:
- recommender training/evaluation
- purchase propensity training and offline policy evaluation

Flow: Bronze (`data/bronze/online_retail_ii/raw.csv`) → Silver (`data/silver/transactions_line_items/transactions_line_items.csv`) → Gold (`data/gold/feature_store/recommender/...` and `data/gold/feature_store/purchase_propensity/...`).

Build commands (separable by layer/engine):

```bash
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_shared_silver --shared-config configs/shared.yaml
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_gold_purchase_propensity --config configs/purchase_propensity/cycle_initial.yaml
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_gold_recommender --output-root data
```

Each command runs transformations, executes DQ checks, and writes a run manifest for its scope.

Purchase propensity gold outputs now materialize:
- labels for `30d`, `60d`, `90d` horizons in `gold_labels`
- feature rollups for `30d`, `60d`, `90d`, `120d` windows in `gold_user_features_asof`
- a training dataset that includes the above labels/features in one row-per-user snapshot

Build entrypoints:

| Command | Scope |
| --- | --- |
| `build_shared_silver` | Shared silver layer from raw source |
| `build_gold_purchase_propensity` | Purchase-propensity gold tables for strict 12-month cycle panel from shared silver |
| `build_gold_recommender` | Recommender gold tables from shared silver |

Common arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `--panel-end-date YYYY-MM-DD` | from purchase config | Purchase-propensity anchor date; builder derives previous 11 monthly snapshots + anchor (strict 12 total). |
| `--recommender-min-event-date YYYY-MM-DD` | unset | Optional lower bound event_date filter for recommender outputs. |
| `--recommender-max-event-date YYYY-MM-DD` | unset | Optional upper bound event_date filter for recommender outputs. |
| `--split-version STRING` | `time_rank_v1` | Writes split-strategy/version tag into `gold_user_item_splits.split_version` and run manifest. |
| `--input-csv PATH` | `data/bronze/online_retail_ii/raw.csv` | Overrides raw input CSV path. |
| `--output-root PATH` | `data` | Overrides root output directory for materialized artifacts. |
| `--bad-ts-threshold FLOAT` | `0.01` | Fails build if bad timestamp ratio in raw input exceeds threshold. |

Run manifest captures build lineage (inputs, params, quality, artifacts) for reproducibility, debugging, and downstream experiment tracking.

Why `--panel-end-date` matters:

- Purchase propensity requires strict 12 monthly snapshots (`10 train / 1 validation / 1 test`) derived from one anchor.
- Example: `--panel-end-date 2010-11-01` materializes partitions from `2009-12-01` through `2010-11-01`.
- Changing the anchor shifts the full rolling panel while preserving strict split shape for consistent backtests.

Key references:

| Purpose | Source of truth |
| --- | --- |
| Schema (columns, PK, types, rules) | `docs/feature_store/data_dictionary.md` |
| Target data model (lineage, layers, engine boundaries) | `docs/feature_store/diagrams/feature_store_target_data_model.mmd` |
| Data quality checks and failure rules | `docs/feature_store/dq_spec.md` |
| Diagram tooling instructions (render workflow only) | `docs/feature_store/diagrams/README.md` |
