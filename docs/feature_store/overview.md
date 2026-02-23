# Feature Store Overview

## Purpose

Provide reproducible, point-in-time datasets for:

- Growth uplift modeling
- Recommender training/evaluation

Raw source is stored in Bronze and transformed into Silver/Gold using DuckDB SQL.

---

## Canonical Docs

- Diagram source: `docs/feature_store/diagrams/feature_store_target_model.mmd`
- Data dictionary (field-level contract): `docs/feature_store/data_dictionary.md`
- Data quality rules: `docs/feature_store/dq.md`

Use this file for high-level context only.

---

## Pipeline Flow

1. Bronze: `data/bronze/online_retail_ii/raw.csv`
2. Silver: `data/silver/transactions_line_items/transactions_line_items.csv`
3. Gold:
   - `data/gold/feature_store/interaction_events/interaction_events.csv`
   - `data/gold/feature_store/user_item_splits/user_item_splits.csv`
   - `data/gold/feature_store/labels/as_of_date=<DATE>/labels.csv`
   - `data/gold/feature_store/user_features_asof/as_of_date=<DATE>/user_features_asof.csv`
   - `data/gold/feature_store/uplift_train_dataset/as_of_date=<DATE>/uplift_train_dataset.csv`
   - `data/gold/feature_store/_meta/as_of_date=<DATE>/run_manifest.json`

---

## Build

Run:

```bash
PYTHONPATH=src .venv/bin/python -m mle_marketplace_growth.feature_store.build
```

Key options:

- `--input-csv` (default: `data/bronze/online_retail_ii/raw.csv`)
- `--as-of-date YYYY-MM-DD` (default: max source event date)
- `--split-version` (default: `time_rank_v1`)
- `--output-root` (default: `data`)
- `--bad-ts-threshold` (default: `0.01`)

---

## Diagram Workflow

- Edit Mermaid source in `docs/feature_store/diagrams/feature_store_target_model.mmd`
- Render assets with `scripts/render_diagrams.sh` (optional for viewers; required only when diagram source changes)
- Ready commands are listed in `docs/feature_store/diagrams/README.md`
