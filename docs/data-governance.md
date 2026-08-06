# Data Governance

Phase 1 is a source and licensing audit only. The registry is metadata and policy evidence; it is not permission to download, ingest, redistribute, or train on any source.

## Evidence hierarchy

Evidence is ranked as: (1) source-specific licence or terms from the rights holder; (2) official dataset card or government open-data policy that explicitly applies to the named dataset; (3) repository licence files and release metadata; (4) official paper or institutional description; and (5) discovery leads. A paper, repository, public URL, or government authority establishes provenance or context only unless it explicitly grants the relevant right.

## Rights vocabulary

Paper rights cover the paper. Code rights cover code. Dataset rights cover the dataset and its annotations. Original-source rights cover the underlying legislation, rulings, or records. These rights are recorded separately because one does not transfer automatically to another.

## Permission states and decisions

Every permission uses `yes`, `no`, `conditional`, or `unknown`. `yes` requires an explicit evidence URL. `conditional` requires conditions. Unresolved or conflicting evidence cannot produce `approved`; it produces `conditional`, `blocked_pending_review`, `metadata_only`, or `excluded`.

`attribution_required=yes` is narrower: it requires a cited licence or terms page that explicitly requires attribution or preservation of notices. A missing or silent licence/terms record must remain `unknown`, even if another permission has evidence.

An approved record must have resolved licence status, resolved required rights, permitted automated access, no unresolved required rights, and no unresolved high-PII public-demo claim. A blocked record must state a manual next action. Evaluation-only sources cannot become primary corpora.

## Privacy and access

Case-derived material is high privacy risk until a source-specific review documents anonymisation, lawful basis, access controls, and public-demo mitigation. Public visibility is not a privacy clearance. Automated access is independently governed from human web access; restrictive terms block automation.

## Operational boundary

Runtime validation is offline and reads only the local CSV. It never contacts source websites, follows URLs, downloads files, or creates directories. Manual review must resolve the registry action before any future source workflow is designed.
