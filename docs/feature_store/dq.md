# Feature Store Data Quality Checks

Detailed data quality checks for feature-store datasets.

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
| `DQ_GOLD_003` | Gold `labels` | Label metadata validity | `label_name` in `{net_revenue_30d,purchase_30d}` and `window_days=30` | Error | Stop pipeline |
| `DQ_GOLD_004` | Gold `interaction_events` | Non-null interaction keys | `user_id`, `item_id`, `event_ts` non-null | Error | Drop bad rows and report count |
| `DQ_GOLD_005` | Gold `user_item_splits` | Split validity | `split` in `{train,val,test}` and `split_version` non-empty | Error | Stop pipeline |
| `DQ_GOLD_006` | Gold `user_item_splits` | Chronology | `max(train_ts) <= min(val_ts) <= min(test_ts)` | Error | Stop pipeline |
| `DQ_GOLD_007` | Gold `embeddings` | Vector shape consistency | Same embedding dimension for given `model_version` | Error | Stop pipeline |
| `DQ_GOLD_008` | Gold `policies` | Score/treat consistency | If `treat_flag=1`, `score` is non-null | Warn | Report and continue |
| `DQ_GOLD_009` | Gold `uplift_train_dataset` | Unique train grain | Unique `(user_id, as_of_date)` | Error | Stop pipeline |
