# Feature Store Data Quality Checks

Implemented data quality checks and cleaning behavior for the feature-store pipelines that feed recommender and purchase propensity engines.

| Check ID | Dataset | Check | Rule | Severity | Action on Failure |
| --- | --- | --- | --- | --- | --- |
| `DQ_RAW_001` | Raw CSV | Required columns present | All 8 source columns exist | Error | Stop pipeline |
| `DQ_RAW_002` | Raw CSV | Timestamp parseability | `InvoiceDate` parses to timestamp | Error | Drop bad rows and fail if above threshold |
| `DQ_RAW_003` | Raw CSV | Exact duplicate rows count | Count exact duplicates for visibility in manifest | Info | Continue |
| `DQ_GOLD_001` | Gold `user_features_asof` | Unique feature grain | Unique `(user_id, as_of_date)` | Error | Stop pipeline |
| `DQ_GOLD_003` | Gold `labels` | Label metadata validity | `label_name` in `{net_revenue_30d,purchase_30d,net_revenue_60d,purchase_60d,net_revenue_90d,purchase_90d}` and `window_days` matches suffix | Error | Stop pipeline |
| `DQ_GOLD_005` | Gold `user_item_splits` | Split validity | `split` in `{train,val,test}` and `split_version` non-empty | Error | Stop pipeline |
| `DQ_GOLD_006` | Gold `user_item_splits` | Chronology | `max(train_ts) <= min(val_ts) <= min(test_ts)` | Error | Stop pipeline |
| `DQ_GOLD_009` | Gold `propensity_train_dataset` | Unique train grain | Unique `(user_id, as_of_date)` | Error | Stop pipeline |

Silver cleaning summary (`transactions_line_items`):

- Applies typing/validity filters (`InvoiceDate`, `Price`, `Quantity`) and drops invalid rows.
- Removes exact raw duplicates (`SELECT DISTINCT` on source columns).
- Full column-level normalization rules are documented in `docs/feature_store/data_dictionary.md`.
