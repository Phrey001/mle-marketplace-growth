# Data Dictionary

This dictionary defines the current implemented feature-store datasets.
It is the field-level companion to the high-level target data model in `docs/feature_store/overview.md`.

Physical models live in `src/mle_marketplace_growth/feature_store/sql/`; outputs are materialized to `data/silver/` and `data/gold/feature_store/<engine>/`.

---

## `silver_transactions_line_items`

Layer: `Silver`  
Domain: `Shared`  
Purpose: canonical cleaned transaction line items shared by both engines.  

Grain: one row per transaction line item.
Note: exact raw duplicate rows are removed before typing/filters.

| Column | PK | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- | --- |
| `invoice_id` | Yes | string | No | Invoice identifier | `Invoice` trimmed | `489434` |
| `item_id` | Yes | string | No | Product identifier | `StockCode` trimmed | `85048` |
| `item_description` | No | string | Yes | Product description | `Description` trimmed; blank normalized to `NULL` | `15CM CHRISTMAS GLASS BALL 20 LIGHTS` |
| `quantity` | No | int | No | Purchased quantity (can be negative for returns/cancellations) | cast from `Quantity`; filtered `<> 0` | `12` |
| `event_ts` | Yes | timestamp | No | Transaction event timestamp | parsed from `InvoiceDate` | `2009-12-01 07:45:00` |
| `event_date` | No | date | No | Date component of event | `CAST(event_ts AS DATE)` | `2009-12-01` |
| `unit_price` | No | double | No | Unit price | cast from `Price`; filtered `>= 0` | `6.95` |
| `line_revenue` | No | double | No | Line revenue | `quantity * unit_price` | `83.4` |
| `user_id` | No | string | Yes | Normalized customer id | `Customer ID` trimmed; strips trailing `.0` when numeric; may be blank in Silver | `13085` |
| `country` | No | string | Yes | Customer country | `Country` trimmed | `United Kingdom` |

---

## `gold_interaction_events`

Layer: `Gold`  
Domain: `Recommender`  
Purpose: positive user-item interaction events for recommender training and split derivation.  

Grain: one row per positive user-item interaction event.
Note: current recommender ML starts from binary user-item interactions only (each unique pair counted once; duplicates collapsed; quantity ignored). Popularity then applies log-scaled item interaction counts, MF applies TF-IDF weighting to the binary interaction matrix, and two-tower uses the binary pairs directly. `weight` is retained for future extensions and is not used as a direct training weight today.

| Column | PK | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- | --- |
| `user_id` | Yes | string | No | User identifier | from silver; filtered non-empty | `13085` |
| `item_id` | Yes | string | No | Item identifier | from silver | `21232` |
| `invoice_id` | Yes | string | No | Invoice identifier | from silver | `489434` |
| `event_ts` | Yes | string (timestamp format) | No | Event timestamp in `%Y-%m-%d %H:%M:%S` | formatted from silver `event_ts` | `2009-12-01 07:45:00` |
| `event_date` | No | string (date format) | No | Event date in `%Y-%m-%d` | cast from silver `event_date` | `2009-12-01` |
| `weight` | No | int | No | Retained interaction-strength field | silver `quantity`; filtered `> 0`; not used by current recommender ML training path | `24` |

---

## `gold_user_item_splits`

Layer: `Gold`  
Domain: `Recommender`  
Purpose: train/validation/test split assignments for recommender offline evaluation.  

Grain: one row per interaction event with assigned split.
Note: split rows keep only the fields needed for chronological evaluation and downstream indexing; retained interaction `weight` stays only in `gold_interaction_events`.

| Column | PK | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- | --- |
| `user_id` | Yes | string | No | User identifier | from `gold_interaction_events` | `13085` |
| `item_id` | Yes | string | No | Item identifier | from `gold_interaction_events` | `21232` |
| `invoice_id` | Yes | string | No | Invoice identifier | from `gold_interaction_events` | `489434` |
| `event_ts` | Yes | string (timestamp format) | No | Event timestamp | from `gold_interaction_events` | `2009-12-01 07:45:00` |
| `event_date` | No | string (date format) | No | Event date | from `gold_interaction_events` | `2009-12-01` |
| `split` | Yes | enum(`train`,`val`,`test`) | No | Time-based split assignment per user invoice moment | latest invoice moment=`test`, second-latest=`val`, else=`train` | `train` |

---

## `gold_recommender_user_index`

Layer: `Gold`  
Domain: `Recommender`  
Purpose: stable user row indices for recommender matrix and embedding models.  

Grain: one row per recommender user id.

| Column | PK | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- | --- |
| `user_id` | Yes | string | No | User identifier | distinct from `gold_user_item_splits` | `13085` |
| `user_idx` | No | int | No | Stable zero-based user row index for recommender matrices | `ROW_NUMBER() OVER (ORDER BY user_id) - 1` | `0` |

---

## `gold_recommender_item_index`

Layer: `Gold`  
Domain: `Recommender`  
Purpose: stable train-item row indices for recommender matrix and embedding models.  

Grain: one row per recommender train-split item id.

| Column | PK | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- | --- |
| `item_id` | Yes | string | No | Item identifier | distinct train-split items from `gold_user_item_splits` | `21232` |
| `item_idx` | No | int | No | Stable zero-based item row index for recommender matrices | `ROW_NUMBER() OVER (ORDER BY item_id) - 1` | `0` |

---

## `gold_user_features_asof`

Layer: `Gold`  
Domain: `Purchase Propensity`  
Purpose: point-in-time user feature snapshots for propensity training and scoring.  

Grain: one row per user per as-of date.

| Column | PK | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- | --- |
| `user_id` | Yes | string | No | User identifier | distinct users from events up to `as_of_date` | `12347` |
| `as_of_date` | Yes | date | No | Feature snapshot date | CLI arg or max event date | `2011-12-09` |
| `recency_days` | No | int | No | Days since latest positive event | `date_diff(as_of_date - last_event_date)`; default `9999` | `2` |
| `frequency_30d` | No | int | No | Distinct invoice count in trailing 30 days | from positive events | `1` |
| `frequency_60d` | No | int | No | Distinct invoice count in trailing 60 days | from positive events | `1` |
| `frequency_90d` | No | int | No | Distinct invoice count in trailing 90 days | from positive events | `2` |
| `frequency_120d` | No | int | No | Distinct invoice count in trailing 120 days | from positive events | `3` |
| `monetary_30d` | No | double | No | Revenue in trailing 30 days | sum positive `line_revenue` | `224.82` |
| `monetary_60d` | No | double | No | Revenue in trailing 60 days | sum positive `line_revenue` | `622.15` |
| `monetary_90d` | No | double | No | Revenue in trailing 90 days | sum positive `line_revenue` | `1519.14` |
| `monetary_120d` | No | double | No | Revenue in trailing 120 days | sum positive `line_revenue` | `1780.44` |
| `avg_basket_value_60d` | No | double | No | Average invoice revenue in trailing 60 days | average of invoice totals | `311.08` |
| `avg_basket_value_90d` | No | double | No | Average invoice revenue in trailing 90 days | average of invoice totals | `759.57` |
| `avg_basket_value_120d` | No | double | No | Average invoice revenue in trailing 120 days | average of invoice totals | `593.48` |
| `country` | No | string | No | Latest known country as of date | `arg_max(country, event_ts)` with default `UNKNOWN` | `Iceland` |

---

## `gold_labels`

Layer: `Gold`  
Domain: `Purchase Propensity`  
Purpose: future-window purchase and revenue outcomes per user snapshot.  

Grain: one row per (`user_id`, `as_of_date`, `label_name`).

Note: each (`user_id`, `as_of_date`, `window_days`) has multiple rows (one per label type: purchase and net_revenue).

| Column | PK | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- | --- |
| `user_id` | Yes | string | No | User identifier | distinct users up to `as_of_date` | `12347` |
| `as_of_date` | Yes | date | No | Label snapshot anchor date | CLI arg or max event date | `2011-11-09` |
| `label_name` | Yes | enum(`net_revenue_30d`,`purchase_30d`,`net_revenue_60d`,`purchase_60d`,`net_revenue_90d`,`purchase_90d`) | No | Label identifier | generated by label model | `net_revenue_60d` |
| `label_value` | No | double | No | Label numeric value | future window aggregation per `label_name` | `224.82` |
| `window_days` | No | int | No | Future label horizon | one of `30`, `60`, `90` | `60` |

---

## `gold_propensity_train_dataset`

Layer: `Gold`  
Domain: `Purchase Propensity`  
Purpose: model-ready joined feature-plus-label dataset for propensity ML runs.  

Grain: one row per (`user_id`, `as_of_date`) for propensity training/evaluation.
Note: dataset includes 30/60/90 labels and multi-lookback features; current main training target remains `label_purchase_30d`.

| Column | PK | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- | --- |
| `user_id` | Yes | string | No | User identifier | from `gold_user_features_asof` | `12347` |
| `as_of_date` | Yes | date | No | Training anchor date | from `gold_user_features_asof` | `2011-11-09` |
| `recency_days` | No | int | No | Recency feature | from `gold_user_features_asof` | `9` |
| `frequency_30d` | No | int | No | Frequency feature (30d) | from `gold_user_features_asof` | `1` |
| `frequency_60d` | No | int | No | Frequency feature (60d) | from `gold_user_features_asof` | `1` |
| `frequency_90d` | No | int | No | Frequency feature (90d) | from `gold_user_features_asof` | `1` |
| `frequency_120d` | No | int | No | Frequency feature (120d) | from `gold_user_features_asof` | `2` |
| `monetary_30d` | No | double | No | Monetary feature (30d) | from `gold_user_features_asof` | `1294.32` |
| `monetary_60d` | No | double | No | Monetary feature (60d) | from `gold_user_features_asof` | `1294.32` |
| `monetary_90d` | No | double | No | Monetary feature (90d) | from `gold_user_features_asof` | `1294.32` |
| `monetary_120d` | No | double | No | Monetary feature (120d) | from `gold_user_features_asof` | `1294.32` |
| `avg_basket_value_60d` | No | double | No | Basket-value feature (60d) | from `gold_user_features_asof` | `1294.32` |
| `avg_basket_value_90d` | No | double | No | Basket-value feature (90d) | from `gold_user_features_asof` | `1294.32` |
| `avg_basket_value_120d` | No | double | No | Basket-value feature (120d) | from `gold_user_features_asof` | `1294.32` |
| `country` | No | string | No | User country snapshot | from `gold_user_features_asof` | `Iceland` |
| `label_net_revenue_60d` | No | double | No | Revenue label in next 60 days | pivoted from `gold_labels` | `328.55` |
| `label_net_revenue_90d` | No | double | No | Revenue label in next 90 days | pivoted from `gold_labels` | `510.21` |
| `label_net_revenue_30d` | No | double | No | Revenue label in next 30 days | pivoted from `gold_labels` | `224.82` |
| `label_purchase_60d` | No | double | No | Binary purchase label in next 60 days | pivoted from `gold_labels` | `1.0` |
| `label_purchase_90d` | No | double | No | Binary purchase label in next 90 days | pivoted from `gold_labels` | `1.0` |
| `label_purchase_30d` | No | double | No | Binary purchase label in next 30 days | pivoted from `gold_labels` | `1.0` |
