# Data Dictionary

This dictionary defines the current implemented feature-store datasets.
It is the field-level companion to the high-level target data model in `docs/feature_store/overview.md`.

---

## `silver_transactions_line_items`

Grain: one row per transaction line item.
Primary key (practical): (`invoice_id`, `item_id`, `event_ts`)  

| Column | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- |
| `invoice_id` | string | No | Invoice identifier | `Invoice` trimmed | `489434` |
| `item_id` | string | No | Product identifier | `StockCode` trimmed | `85048` |
| `item_description` | string | Yes | Product description | `Description` trimmed | `15CM CHRISTMAS GLASS BALL 20 LIGHTS` |
| `quantity` | int | No | Purchased quantity (can be negative in source) | cast from `Quantity` | `12` |
| `event_ts` | timestamp | No | Transaction event timestamp | parsed from `InvoiceDate` | `2009-12-01 07:45:00` |
| `event_date` | date | No | Date component of event | `CAST(event_ts AS DATE)` | `2009-12-01` |
| `unit_price` | double | No | Unit price | cast from `Price`; filtered `>= 0` | `6.95` |
| `line_revenue` | double | No | Line revenue | `quantity * unit_price` | `83.4` |
| `user_id` | string | Yes | Normalized customer id | `Customer ID` trimmed; strips trailing `.0` when numeric | `13085` |
| `country` | string | Yes | Customer country | `Country` trimmed | `United Kingdom` |

---

## `gold_interaction_events`

Grain: one row per positive user-item interaction event.
Primary key (practical): (`user_id`, `item_id`, `event_ts`, `invoice_id`)

| Column | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- |
| `user_id` | string | No | User identifier | from silver; filtered non-empty | `13085` |
| `item_id` | string | No | Item identifier | from silver | `21232` |
| `invoice_id` | string | No | Invoice identifier | from silver | `489434` |
| `event_ts` | string (timestamp format) | No | Event timestamp in `%Y-%m-%d %H:%M:%S` | formatted from silver `event_ts` | `2009-12-01 07:45:00` |
| `event_date` | string (date format) | No | Event date in `%Y-%m-%d` | cast from silver `event_date` | `2009-12-01` |
| `weight` | int | No | Interaction weight | silver `quantity`; filtered `> 0` | `24` |

---

## `gold_user_item_splits`

Grain: one row per interaction event with assigned split.
Primary key (practical): (`user_id`, `item_id`, `event_ts`, `invoice_id`, `split_version`, `split`)

| Column | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- |
| `user_id` | string | No | User identifier | from `gold_interaction_events` | `13085` |
| `item_id` | string | No | Item identifier | from `gold_interaction_events` | `21232` |
| `invoice_id` | string | No | Invoice identifier | from `gold_interaction_events` | `489434` |
| `event_ts` | string (timestamp format) | No | Event timestamp | from `gold_interaction_events` | `2009-12-01 07:45:00` |
| `event_date` | string (date format) | No | Event date | from `gold_interaction_events` | `2009-12-01` |
| `weight` | int | No | Interaction weight | from `gold_interaction_events` | `24` |
| `split_version` | string | No | Version identifier for split strategy | CLI `--split-version` | `time_rank_v1` |
| `split` | enum(`train`,`val`,`test`) | No | Time-based split assignment per user | latest=`test`, second-latest=`val`, else=`train` | `train` |

---

## `gold_user_features_asof`

Grain: one row per user per as-of date.
Primary key: (`user_id`, `as_of_date`)

| Column | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- |
| `user_id` | string | No | User identifier | distinct users from events up to `as_of_date` | `12347` |
| `as_of_date` | date | No | Feature snapshot date | CLI arg or max event date | `2011-12-09` |
| `recency_days` | int | No | Days since latest positive event | `date_diff(as_of_date - last_event_date)`; default `9999` | `2` |
| `frequency_30d` | int | No | Distinct invoice count in trailing 30 days | from positive events | `1` |
| `frequency_90d` | int | No | Distinct invoice count in trailing 90 days | from positive events | `2` |
| `monetary_30d` | double | No | Revenue in trailing 30 days | sum positive `line_revenue` | `224.82` |
| `monetary_90d` | double | No | Revenue in trailing 90 days | sum positive `line_revenue` | `1519.14` |
| `avg_basket_value_90d` | double | No | Average invoice revenue in trailing 90 days | average of invoice totals | `759.57` |
| `country` | string | No | Latest known country as of date | `arg_max(country, event_ts)` with default `UNKNOWN` | `Iceland` |

---

## `gold_labels`

Grain: one row per (`user_id`, `as_of_date`, `label_name`).
Primary key: (`user_id`, `as_of_date`, `label_name`)

| Column | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- |
| `user_id` | string | No | User identifier | distinct users up to `as_of_date` | `12347` |
| `as_of_date` | date | No | Label snapshot anchor date | CLI arg or max event date | `2011-11-09` |
| `label_name` | enum(`net_revenue_30d`,`purchase_30d`) | No | Label identifier | generated by label model | `net_revenue_30d` |
| `label_value` | double | No | Label numeric value | future window aggregation per `label_name` | `224.82` |
| `window_days` | int | No | Future label horizon | fixed `30` | `30` |

---

## `gold_uplift_train_dataset`

Grain: one row per (`user_id`, `as_of_date`) for uplift training/evaluation.
Primary key: (`user_id`, `as_of_date`)

| Column | Type | Nullable | Description | Source / Rule | Example |
| --- | --- | --- | --- | --- | --- |
| `user_id` | string | No | User identifier | from `gold_user_features_asof` | `12347` |
| `as_of_date` | date | No | Training anchor date | from `gold_user_features_asof` | `2011-11-09` |
| `recency_days` | int | No | Recency feature | from `gold_user_features_asof` | `9` |
| `frequency_30d` | int | No | Frequency feature (30d) | from `gold_user_features_asof` | `1` |
| `frequency_90d` | int | No | Frequency feature (90d) | from `gold_user_features_asof` | `1` |
| `monetary_30d` | double | No | Monetary feature (30d) | from `gold_user_features_asof` | `1294.32` |
| `monetary_90d` | double | No | Monetary feature (90d) | from `gold_user_features_asof` | `1294.32` |
| `avg_basket_value_90d` | double | No | Basket-value feature (90d) | from `gold_user_features_asof` | `1294.32` |
| `country` | string | No | User country snapshot | from `gold_user_features_asof` | `Iceland` |
| `label_net_revenue_30d` | double | No | Revenue label in next 30 days | pivoted from `gold_labels` | `224.82` |
| `label_purchase_30d` | double | No | Binary purchase label in next 30 days | pivoted from `gold_labels` | `1.0` |
