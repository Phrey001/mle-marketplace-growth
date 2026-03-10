# Feature Store Overview

Purpose: produce reproducible point-in-time datasets for:
- recommender training/evaluation
- purchase propensity training and offline policy evaluation

Flow: Bronze (`data/bronze/online_retail_ii/raw.csv`) -> Silver (`data/silver/transactions_line_items/transactions_line_items.parquet`, canonicalized in `data/_tmp/feature_store.duckdb`) -> Gold (`data/gold/feature_store/recommender/...` and `data/gold/feature_store/purchase_propensity/...`).

Silver canonicalization behavior:
- exact raw duplicates are removed first
- rows are typed/validated
- final silver is aggregated to unique grain `(invoice_id, item_id, event_ts)` (true silver key)

Build commands (separable by layer/engine):

```bash
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_shared_silver --shared-config configs/shared.yaml
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_gold_purchase_propensity --config configs/purchase_propensity/cycle_initial.yaml
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build_gold_recommender --config configs/recommender/default.yaml --shared-config configs/shared.yaml
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

Common arguments (shared + recommender builders):

| Argument | Default | Purpose |
| --- | --- | --- |
| `--shared-config PATH` | optional | Shared config defaults (for example `output_root`). |
| `--config PATH` | required (gold scripts) | Engine-specific config defaults. |
| `--output-root PATH` | from config (`data`) | Root output directory for materialized artifacts. |
| `--recommender-min-event-date YYYY-MM-DD` | unset | Optional lower bound event_date filter for recommender outputs. |
| `--recommender-max-event-date YYYY-MM-DD` | unset | Optional upper bound event_date filter for recommender outputs. |

Purchase propensity builder uses config-only inputs:

| Argument | Default | Purpose |
| --- | --- | --- |
| `--config PATH` | required | Purchase-propensity engine config (includes `panel_end_date`, `output_root`). |

Run manifest captures build lineage (inputs, params, quality, artifacts) for reproducibility, debugging, and downstream experiment tracking.

Note: recommender split strategy is fixed in this repo (`split_version='time_rank_v1'`), so it is no longer exposed as a config/CLI knob.

Why `panel_end_date` matters:

- Purchase propensity requires strict 12 monthly snapshots (`10 train / 1 validation / 1 test`) derived from one anchor.
- Example: `panel_end_date: 2010-11-01` materializes partitions from `2009-12-01` through `2010-11-01`.
- Changing the anchor shifts the full rolling panel while preserving strict split shape for consistent backtests.

Key references:

| Purpose | Source of truth |
| --- | --- |
| Schema (columns, PK, types, rules) | `docs/feature_store/data_dictionary.md` |
| Target data model (lineage, layers, engine boundaries) | `docs/feature_store/diagrams/feature_store_target_data_model.mmd` |
| Data quality checks and failure rules | `docs/feature_store/dq_spec.md` |
| Diagram tooling instructions (render workflow only) | `docs/feature_store/diagrams/README.md` |

## Feature Store Data Debugging

Use these exports for ad hoc inspection and debugging. CSVs are written next to the input parquet.

Example: export a single parquet to CSV:

```bash
PYTHONPATH=src python scripts/read_parquet.py \
  data/gold/feature_store/purchase_propensity/labels/as_of_date=2010-11-01/labels.parquet
```

Export latest partitions (current silver + purchase propensity gold only):

```bash
# Latest silver
PYTHONPATH=src python scripts/read_parquet.py \
  "$(ls -d data/silver/transactions_line_items/transactions_line_items.parquet | tail -1)"

# Latest gold labels
PYTHONPATH=src python scripts/read_parquet.py \
  "$(ls -d data/gold/feature_store/purchase_propensity/labels/as_of_date=*/labels.parquet | sort | tail -1)"

# Latest gold user_features_asof
PYTHONPATH=src python scripts/read_parquet.py \
  "$(ls -d data/gold/feature_store/purchase_propensity/user_features_asof/as_of_date=*/user_features_asof.parquet | sort | tail -1)"

# Latest gold propensity_train_dataset
PYTHONPATH=src python scripts/read_parquet.py \
  "$(ls -d data/gold/feature_store/purchase_propensity/propensity_train_dataset/as_of_date=*/propensity_train_dataset.parquet | sort | tail -1)"
```
