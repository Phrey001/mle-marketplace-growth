# Feature Store Layout

This repo uses **Online Retail II** as the single raw source dataset:

- Raw CSV (current location): `data/online_retail_II.csv/online_retail_II.csv`
- Raw zip (optional): `data/online_retail_II.csv.zip`

The feature store is organized using a lightweight **medallion** layout (bronze/silver/gold) to support two independent engines:

- **Growth Uplift** (growth decisioning)
- **Retrieval & Ranking** (content/product selection)

---

## Source Schema (Online Retail II)

Columns in `online_retail_II.csv`:

- `Invoice` (invoice id)
- `StockCode` (item id)
- `Description` (item name/description)
- `Quantity` (units; can be negative for returns/cancellations in many retail datasets)
- `InvoiceDate` (timestamp)
- `Price` (unit price)
- `Customer ID` (user id; may be missing)
- `Country`

Canonical names used in downstream tables:

- `invoice_id`, `item_id`, `item_description`, `quantity`, `event_ts`, `unit_price`, `user_id`, `country`

---

## Directory Layout

Keep the existing raw dataset where it is; the layout below describes *derived* tables (and an optional future move for raw data).

- `data/bronze/`
  - (Optional) `online_retail_ii/raw.csv` (copy/symlink of the raw CSV if you want a single convention)
- `data/silver/`
  - `transactions_line_items/` — cleaned line items (typed columns, normalized ids, consistent timestamps)
  - `invoices/` — invoice-level aggregates (e.g., invoice total, item count)
  - `items/` — item dimension (id, description)
  - `users/` — user dimension (id, country, first/last seen)
- `data/gold/feature_store/`
  - `user_features_asof/` — user-level RFM + behavioural aggregates, computed “as-of” a decision timestamp
  - `interaction_events/` — recommender interactions: `(user_id, item_id, event_ts, weight)`
  - `user_item_splits/` — time-based split tables for training/validation/test
  - `labels/` — uplift labels (e.g., net revenue in a post-assignment window)
  - `embeddings/` — item embeddings snapshots (by `model_version`, `as_of_date`)
  - `policies/` — uplift policies / treatment decisions (by `policy_version`, `as_of_date`)

---

## Table “Contracts” (What each engine consumes)

### Growth Uplift (Decisioning)

Inputs:

- `user_features_asof(as_of_ts)` (RFM + aggregates; built only from events `<= as_of_ts`)
- `labels` for evaluation/training (e.g., `net_revenue_{window}` computed from events `(as_of_ts, as_of_ts + window]`)

Outputs:

- `policies` (who to treat, and optional treatment/budget metadata)
- offline policy evaluation report comparing baseline vs uplift policy

### Retrieval & Ranking (Selection)

Inputs:

- `interaction_events` derived from transactions line items
- `user_item_splits` (time-based split)
- optional `embeddings/item_embeddings` (precomputed item embeddings for serving/retrieval)

Outputs:

- top-K candidates per user (offline) and retrieval metrics (recall/NDCG/hit rate)

---

## Leakage / “As-Of” Rules (Must-Haves)

- Every gold feature row has an `as_of_ts` (or `as_of_date`) and is computed using only source data with timestamps `<= as_of_ts`.
- Labels are computed strictly *after* `as_of_ts` (e.g., a 30-day post-assignment window).
- For recommender evaluation, splits are time-based and respect chronology (no future interactions in train).

