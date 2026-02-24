# Feature Store Overview

Purpose: produce reproducible point-in-time datasets for recommender and growth-uplift training.

Flow: Bronze (`data/bronze/online_retail_ii/raw.csv`) → Silver (`data/silver/transactions_line_items/transactions_line_items.csv`) → Gold (`data/gold/feature_store/recommender/...` and `data/gold/feature_store/growth_uplift/...`).

Build command:

```bash
PYTHONPATH=src python -m mle_marketplace_growth.feature_store.build
```

This single build command runs transformations, executes DQ checks, and writes the run manifest.

Optional build arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `--as-of-date YYYY-MM-DD` | auto (max source event date) | Fixes point-in-time snapshot date for uplift datasets (`gold_labels`, `gold_user_features_asof`, `gold_uplift_train_dataset`). |
| `--split-version STRING` | `time_rank_v1` | Writes split-strategy/version tag into `gold_user_item_splits.split_version` and run manifest. |
| `--input-csv PATH` | `data/bronze/online_retail_ii/raw.csv` | Overrides raw input CSV path. |
| `--output-root PATH` | `data` | Overrides root output directory for materialized artifacts. |
| `--bad-ts-threshold FLOAT` | `0.01` | Fails build if bad timestamp ratio in raw input exceeds threshold. |

Run manifest captures build lineage (inputs, params, quality, artifacts) for reproducibility, debugging, and downstream experiment tracking.

Why `--as-of-date` matters:

- Feature store should represent “what was known at that time,” not future data.
- Example: running with `--as-of-date 2011-11-09` writes growth-uplift artifacts under `data/gold/feature_store/growth_uplift/*/as_of_date=2011-11-09/`.
- Running again with another date (for example `2011-12-09`) creates a separate snapshot folder, so backtests/reruns can compare model behavior across time slices consistently.
- For uplift training quality, avoid using the latest date as `as_of_date` because the future 30-day label window may be empty.

Key references:

| Purpose | Source of truth |
| --- | --- |
| Schema (columns, PK, types, rules) | `docs/feature_store/data_dictionary.md` |
| Target data model (lineage, layers, engine boundaries) | `docs/feature_store/diagrams/feature_store_target_data_model.mmd` |
| Data quality checks and failure rules | `docs/feature_store/dq_spec.md` |
| Diagram tooling instructions (render workflow only) | `docs/feature_store/diagrams/README.md` |
