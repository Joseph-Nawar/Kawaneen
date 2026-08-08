# Phase 2 privacy inspection

Privacy screening is a local technical signal, not legal clearance. It scans acquired CSV and Parquet rows for emails, phone-like values, IBAN-like values, context-supported identity numbers, passport/address indicators, and identifier-like columns. Findings contain masked values only and are written to ignored `artifacts/private/` review bundles.

The screen is deterministic and records detector, row location, column, and a masked finding. It does not rewrite source data, remove records, infer consent, or establish that a dataset is safe to publish, quote, train on, or demonstrate publicly. A source remains pending manual review until original-source rights, anonymisation, jurisdictional privacy requirements, and proposed use are separately assessed.

The privacy status manifest records the finding count and `legal_clearance=false`. Review bundles must never be staged.
