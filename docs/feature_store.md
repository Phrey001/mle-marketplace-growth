# Feature Store Layout

This repo uses **Online Retail II** as the single raw source dataset:

- Raw CSV (current location): `data/online_retail_II.csv/online_retail_II.csv`

The feature store is organized using a lightweight **medallion** layout (bronze/silver/gold) to support two independent engines:

- **Growth Uplift** (growth decisioning)
- **Retrieval & Ranking** (content/product selection)

---

## Source Schema (Online Retail II)

| Source Column | Canonical Column | Type (target) | Nullable | Notes |
| --- | --- | --- | --- | --- |
| `Invoice` | `invoice_id` | `string` | No | Invoice identifier |
| `StockCode` | `item_id` | `string` | No | Product identifier |
| `Description` | `item_description` | `string` | Yes | Free text description |
| `Quantity` | `quantity` | `int` | No | May be negative for returns/cancellations |
| `InvoiceDate` | `event_ts` | `timestamp` | No | Event timestamp |
| `Price` | `unit_price` | `decimal` | No | Unit price |
| `Customer ID` | `user_id` | `string` | Yes | Missing values should be dropped or bucketed for modeling |
| `Country` | `country` | `string` | Yes | Country dimension |

---

## Directory Layout

| Layer | Path | Purpose |
| --- | --- | --- |
| Bronze | `data/online_retail_II.csv/online_retail_II.csv` | Raw immutable source extract |
| Silver | `data/silver/transactions_line_items/` | Cleaned typed line-item transactions |
| Silver | `data/silver/invoices/` | Invoice-level aggregates (total amount, item count) |
| Silver | `data/silver/items/` | Item dimension |
| Silver | `data/silver/users/` | User dimension (first/last seen, country) |
| Gold | `data/gold/feature_store/user_features_asof/` | Uplift user features at point-in-time |
| Gold | `data/gold/feature_store/interaction_events/` | Recommender interaction events |
| Gold | `data/gold/feature_store/user_item_splits/` | Time-based train/val/test splits |
| Gold | `data/gold/feature_store/labels/` | Post-assignment uplift labels |
| Gold | `data/gold/feature_store/embeddings/` | Item embedding snapshots |
| Gold | `data/gold/feature_store/policies/` | Uplift treatment decisions |

---

## Gold Table Contracts

| Table | Grain / Key | Partition | Main Columns | Consumers |
| --- | --- | --- | --- | --- |
| `user_features_asof` | `(user_id, as_of_date)` | `as_of_date` | `user_id`, `as_of_date`, RFM and behavioral features | Growth Uplift |
| `labels` | `(user_id, as_of_date, label_name)` | `as_of_date` | `label_name`, `label_value`, `window_days` | Growth Uplift |
| `interaction_events` | event-level | `event_date` | `user_id`, `item_id`, `event_ts`, `weight` | Recommender |
| `user_item_splits` | `(user_id, item_id, split)` | `split_date` | `split`, `event_ts` | Recommender |
| `embeddings` | `(item_id, model_version, as_of_date)` | `as_of_date` | `embedding_vector`, `model_version` | Recommender serving |
| `policies` | `(user_id, policy_version, as_of_date)` | `as_of_date` | `treat_flag`, `score`, `budget_bucket` | Uplift serving/eval |

---

## `user_features_asof` Feature Definitions

| Feature | Definition | Window | Null / Default | Leakage Rule |
| --- | --- | --- | --- | --- |
| `recency_days` | Days since latest transaction before `as_of_ts` | trailing | large sentinel (e.g., 9999) | Use only events `<= as_of_ts` |
| `frequency_30d` | Transaction count in last 30 days | 30d trailing | `0` | Use only events `<= as_of_ts` |
| `frequency_90d` | Transaction count in last 90 days | 90d trailing | `0` | Use only events `<= as_of_ts` |
| `monetary_30d` | Sum of net revenue in last 30 days | 30d trailing | `0.0` | Use only events `<= as_of_ts` |
| `monetary_90d` | Sum of net revenue in last 90 days | 90d trailing | `0.0` | Use only events `<= as_of_ts` |
| `avg_basket_value_90d` | Mean invoice value in last 90 days | 90d trailing | `0.0` | Use only events `<= as_of_ts` |
| `country` | User country from latest known value | latest as-of | `UNKNOWN` | Snapshot as-of `as_of_ts` |

---

## Label Definitions

| Label | Definition | Window | Unit | Usage |
| --- | --- | --- | --- | --- |
| `net_revenue_30d` | Sum of post-assignment net revenue | `(as_of_ts, as_of_ts+30d]` | currency | Uplift training/evaluation |
| `purchase_30d` | Any purchase in post-assignment window | `(as_of_ts, as_of_ts+30d]` | binary | Optional uplift target |

---

## Data Quality Checks

| Check ID | Dataset | Check | Rule | Severity | Action on Failure |
| --- | --- | --- | --- | --- | --- |
| `DQ_RAW_001` | Raw CSV | Required columns present | All 8 source columns exist | Error | Stop pipeline |
| `DQ_RAW_002` | Raw CSV | Timestamp parseability | `InvoiceDate` parses to timestamp | Error | Drop bad rows and fail if above threshold |
| `DQ_SILVER_001` | Silver `transactions_line_items` | Non-null keys | `invoice_id`, `item_id`, `event_ts` are non-null | Error | Stop pipeline |
| `DQ_SILVER_002` | Silver `transactions_line_items` | Numeric validity | `quantity` and `unit_price` are numeric | Error | Drop bad rows and report count |
| `DQ_SILVER_003` | Silver `transactions_line_items` | Non-negative price | `unit_price >= 0` | Error | Drop row |
| `DQ_SILVER_004` | Silver `users` | Unique user key | One row per `user_id` | Error | Deduplicate deterministically and alert |
| `DQ_GOLD_001` | Gold `user_features_asof` | Unique feature grain | Unique `(user_id, as_of_date)` | Error | Stop pipeline |
| `DQ_GOLD_002` | Gold `user_features_asof` | As-of leakage guard | Source max timestamp per row `<= as_of_ts` | Error | Stop pipeline |
| `DQ_GOLD_003` | Gold `labels` | Post-window guard | Label window starts after `as_of_ts` | Error | Stop pipeline |
| `DQ_GOLD_004` | Gold `interaction_events` | Non-null interaction keys | `user_id`, `item_id`, `event_ts` non-null | Error | Drop bad rows and report count |
| `DQ_GOLD_005` | Gold `user_item_splits` | Split validity | `split` in `{train,val,test}` | Error | Stop pipeline |
| `DQ_GOLD_006` | Gold `user_item_splits` | Chronology | `max(train_ts) < min(val_ts) <= min(test_ts)` | Error | Stop pipeline |
| `DQ_GOLD_007` | Gold `embeddings` | Vector shape consistency | Same embedding dimension for given `model_version` | Error | Stop pipeline |
| `DQ_GOLD_008` | Gold `policies` | Score/treat consistency | If `treat_flag=1`, `score` is non-null | Warn | Report and continue |

---

## Leakage / “As-Of” Rules (Must-Haves)

- Every gold feature row has an `as_of_ts` (or `as_of_date`) and is computed using only source data with timestamps `<= as_of_ts`.
- Labels are computed strictly *after* `as_of_ts` (e.g., a 30-day post-assignment window).
- For recommender evaluation, splits are time-based and respect chronology (no future interactions in train).
