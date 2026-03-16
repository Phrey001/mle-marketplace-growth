-- Purpose: Create gold_user_item_splits with time-rank train/val/test splits.
-- Why: Offline retrieval evaluation needs a chronological holdout so the newest behavior
-- is reserved for validation/test and older behavior remains in training.
-- Split unit: invoice-level purchase moments, so all items from the same invoice
-- stay in the same split instead of being separated by line-item tie-breakers.
CREATE OR REPLACE TABLE gold_user_item_splits AS
WITH
invoice_moments AS (
  -- Collapse to one row per user invoice/moment before ranking.
  -- This makes the split follow purchase occasions rather than arbitrary
  -- line-item ordering within the same invoice timestamp.
  SELECT
    user_id,
    invoice_id,
    event_ts
  FROM gold_interaction_events
  GROUP BY user_id, invoice_id, event_ts
),
ordered_invoice_moments AS (
  -- Rank each user's invoice moments from most recent to oldest.
  -- recency_rank = 1 -> latest invoice moment (test)
  -- recency_rank = 2 -> second-latest invoice moment (val)
  -- recency_rank >= 3 -> older invoice history (train)
  SELECT
    user_id,
    invoice_id,
    event_ts,
    ROW_NUMBER() OVER (
      PARTITION BY user_id
      ORDER BY event_ts DESC, invoice_id DESC
    ) AS recency_rank
  FROM invoice_moments
),
split_invoice_moments AS (
  -- Convert the per-user invoice recency rank into the fixed chronological split contract.
  SELECT
    user_id,
    invoice_id,
    event_ts,
    CASE
      WHEN recency_rank = 1 THEN 'test'
      WHEN recency_rank = 2 THEN 'val'
      ELSE 'train'
    END AS split
  FROM ordered_invoice_moments
)
-- Join the split decision back to all line items in the same invoice moment.
SELECT
  events.user_id,
  events.item_id,
  events.invoice_id,
  events.event_ts,
  events.event_date,
  split_invoice_moments.split AS split
FROM gold_interaction_events AS events
INNER JOIN split_invoice_moments
  ON events.user_id = split_invoice_moments.user_id
 AND events.invoice_id = split_invoice_moments.invoice_id
 AND events.event_ts = split_invoice_moments.event_ts
ORDER BY events.event_ts, events.user_id, events.item_id;
