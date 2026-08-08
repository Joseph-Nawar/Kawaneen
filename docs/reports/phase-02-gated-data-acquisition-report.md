# Phase 2 gated data acquisition report

Date: 2026-08-07

## Scope and policy

Phase 2 implements gated local acquisition, immutable raw storage, deterministic integrity checks, exact duplicate analysis, and masked privacy screening. Parsing, OCR, normalization, retrieval, training, APIs, and public display remain out of scope.

The Phase 1 registry authorizes three acquired sources under separate, fail-closed decisions: ALARB for controlled private parsing/evaluation and inspection; ArabiCCR for controlled private local research, parsing, and evaluation; and the Saudi MOJ-derived seed for controlled private local parsing and inspection. Training, publishing, public display, and public-demo operations remain denied for every source. The MOJ-derived seed is not evaluation-approved.

## Sources and acquisition results

| Source | Specification | Result | Phase 3 eligibility |
| --- | --- | --- | --- |
| ALARB | `THIQAH-RD/ALARB`, Hugging Face dataset, revision `e64bfdc867146294a65434c5ca16c2c4c5288ca2`, Apache-2.0, 13,341 records | Acquired locally from the official Hub client after resolving cache links; raw bytes remain ignored | Private local parsing and evaluation authorized by risk decision; training, display, and demo remain denied |
| ArabiCCR | Canonical source: Mendeley Data, DOI `10.17632/np538c95yy.3`, Version 3, CC BY 4.0, 12,806 records | Authorized manual official download followed by local import of the exact `ArabiCCR-dataset.csv`; raw bytes remain ignored | Private local parsing/evaluation authorized after automated inspection; training, display, and demo remain denied |
| `saudi-moj-derived` | `WafaaFraih/saudi-legal-moj`, pinned revision `8b55ef5a666ad773c81086051813582fd14eb466`, uploader-declared CC BY 4.0, 3,185 records | Acquired through the official Hub client; raw bytes remain ignored | Private local parsing seed authorized; not authoritative/gold and not evaluation-approved; training, display, and demo remain denied |

The exact ArabiCCR fallback is:

```text
kawaneen data import-local arabiccr --file /path/to/ArabiCCR-dataset.csv --purpose local_research
```

No blocked, metadata-only, conditional-unresolved, or excluded source was downloaded.

## ALARB integrity record

All paths below are repository-relative to the raw source namespace; raw files are not staged.

| File | Size | SHA-256 |
| --- | ---: | --- |
| `README.md` | 1,834 bytes | `85c913413c6ca894b074e243933dc16aa04d898598ba205d2d56f67dc53c7fd6` |
| `data/train-00000-of-00001.parquet` | 19,133,223 bytes | `16a642aeff40e9c8d050ca99a81f7a1802cdbee64234778d6c5790d20750ae32` |
| `data/test-00000-of-00001.parquet` | 2,145,231 bytes | `0bc425145417e8966345af3324bad9627c8bd0f469559684c1d9c8dda81dbae5` |

The train split contains 12,012 records and the test split 1,329 records. Both Parquet files have the ordered columns `case_facts`, `court_reasoning`, `applicable_laws`, and `verdict`. The schema fingerprint for each is `73e05f2483356932bfc71476f4bc044fa99594afe1db9625aea668d2c1185007`.

Exact duplicate analysis found zero duplicate physical files, zero duplicate rows, and zero train/test exact overlap. No raw file was modified.

## Privacy inspection

The deterministic ALARB screen reported 2,532 masked findings and status `pending_manual_review`; `legal_clearance` remains `false`. The bundle is stored only under ignored `artifacts/private/`. Findings are not legal clearance and no unmasked legal text or identifier is included in this report.

## Verification

The initial offline suite contained 69 passing tests with 85.05% branch-aware coverage. The final Phase 2B suite contains 76 passing tests with 86.87% branch-aware coverage, including statutory metrics, MOJ policy, sanitized privacy aggregation, stable multi-source manifests, stage authorization, and revision-pinned acquisition.

The final verification run records:

- `uv sync --locked --dev`: passed.
- `uv run kawaneen data plan`: passed; ALARB, ArabiCCR, and the MOJ-derived seed plans shown without network access.
- `uv run kawaneen data acquire alarb --purpose evaluation`: passed after a cache-symlink handling correction; three files copied and hashed.
- `uv run kawaneen data verify --source alarb`: passed; 13,341 records, zero exact overlap.
- `uv run kawaneen data verify --source arabiccr`: passed; 12,806 records, zero exact duplicate rows.
- `uv run kawaneen data verify --source saudi-moj-derived`: passed; 3,185 records, zero exact duplicate rows.
- `uv run kawaneen data audit --source alarb`: passed; 2,532 masked findings, pending manual review.
- `uv run kawaneen data audit --source arabiccr`: passed; 31,314 masked findings, pending manual review.
- `uv run kawaneen data audit-statutory saudi-moj-derived`: passed; counts-only statutory quality report.
- `uv run kawaneen data manifest build` for all acquired sources: passed; stable multi-source manifests written.
- `uv run kawaneen data manifest validate`: passed.
- `uv run ruff format --check .`: passed.
- `uv run ruff check .`: passed.
- `uv run pyright`: passed with zero errors.
- `uv run pytest`: initial Phase 2 run passed with 69 tests and 85.05% coverage; the final Phase 2B run passed with 76 tests and 86.87% branch-aware coverage.
- `uv run pre-commit run --all-files`: passed.
- `make check`: passed.
- `git diff --check`: passed before staging; the staged check is run as the final handoff check.
- `git diff --cached --check`: passed for the staged sanitized snapshot.

## Manual actions and next-stage gate

1. Confirm ALARB original court-record rights, inspect privacy findings, and obtain explicit permissions for training, quoting/display, and public demonstration.
2. Inspect ArabiCCR Version 3 for residual identifiers, confirm the claimed anonymisation, and review original Saudi Ministry of Justice rights plus training/display/demo terms. Its exact imported file, hash, and size are recorded below.
3. ALARB and ArabiCCR private local parsing/evaluation are authorized under their separate risk decisions; their training, public-display, and demo restrictions remain. The MOJ-derived seed is authorized as a raw parsing seed only and remains unsuitable as an authoritative/gold article corpus before reconciliation. A separate licensed public-demo corpus is still required; an aggregate open-data portal is not a legal-text corpus.

Public display, training, and redistribution are not authorized by this report. ALARB and ArabiCCR private local parsing/evaluation are authorized under their recorded risk decisions; the remaining ArabiCCR actions are manual anonymisation, original-source, and public-use rights review.

## Final Phase 2 qualification and release audit

### Sanitized privacy summaries

| Source | Findings by detector | Affected records | Findings by column | Review sample | Confirmed PII | Likely false positives | Unresolved categories | Status |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| ALARB | phone: 2,532 | 2,266 | `case_facts`: 2,143; `court_reasoning`: 319; `verdict`: 70 | 100 | `not_reviewed` | `not_reviewed` | phone-pattern findings; original-source privacy; manual PII determination | pending_manual_review |
| ArabiCCR v3 | email: 118; IBAN-like: 4; phone: 31,192 | 12,806 | `EVENTS`: 3,977; `REASONING`: 1,159; `RULING`: 510; `case_number`: 3,721; `case_text`: 5,420; `judgment_date`: 12,806; `judgment_number`: 3,721 | 100 | `not_reviewed` | `not_reviewed` | anonymisation; original-source privacy; manual PII determination | pending_manual_review |

All committed privacy output contains counts, column names, detector names, and `[REDACTED]` values only. Automated screening does not confirm PII or establish legal clearance. Unreviewed PII and false-positive classifications are represented as `not_reviewed`, not numeric zero. The ALARB phone-pattern count is retained as unresolved rather than asserted to be PII or a false positive.

### Integrity and manifest qualification

ALARB remains verified at revision `e64bfdc867146294a65434c5ca16c2c4c5288ca2`, with 13,341 records, zero exact duplicate rows, zero physical duplicate files, and zero train/test overlap. Its three file hashes and sizes are in the schema-version-2 `acquisition_lock.json` and the source/version-qualified raw manifest.

Historical initial-audit state (superseded): ArabiCCR Version 3 was not imported because no user-provided CSV was available at that time. The later Phase 2B qualification records the subsequent authorized import and results. The required command remains:

```text
kawaneen data import-local arabiccr --file <path-to-authorized-ArabiCCR-dataset.csv> --purpose local_research
```

The lock and manifests use deterministic ordering, stable schema versions, repository-relative file paths, source/version identities, and no acquisition timestamps. Identical manifest rebuilds produce identical bytes.

### Final next-stage decisions

The snapshot records separate decisions for legal clearance and each use stage. For ALARB, `legal_clearance=false`, `authorized_for_local_parsing=true`, `authorized_for_evaluation=true`, `authorized_for_training=false`, and `authorized_for_public_display=false`. This is a controlled private-use risk decision after automated inspection, not permission to redistribute, train, quote publicly, or demonstrate.

ArabiCCR local parsing/evaluation status is superseded by the Phase 2B qualification below; public display and training remain denied while rights are unresolved.

### Hybrid corpus decision

The public-demo strategy is hybrid: ALARB and ArabiCCR for real private local research; a separately permissioned human-reviewed real-data gold evaluation set; labelled semi-synthetic query-passage pairs for any later authorized retriever/reranker training; explicitly fictional statutes for public demonstrations; and synthetic safety/abstention stress tests. Synthetic material cannot replace human-reviewed real-data evaluation and cannot be presented as real law. Document-level provenance labels and train/test separation are mandatory.

### Release disposition

The code and sanitized multi-source snapshot pass the automated release checks. Phase 2B qualification is complete for the acquired seeds, subject to the unresolved legal actions recorded below. No raw data or private review bundle is staged.

## Phase 2B — Statutory corpus qualification and seed acquisition

### ArabiCCR retry and qualification

The user-provided CSV was retried through the required command:

```text
kawaneen data import-local arabiccr --file data/raw/arabiccr/3/ArabiCCR-dataset.csv --purpose local_research
```

The import, integrity verification, privacy audit, and manifest build passed. The final raw file’s acquisition provenance is Mendeley Data—not the raw-file path itself. ArabiCCR Version 3 recorded:

- version: `3`; DOI: `10.17632/np538c95yy.3`;
- file: `ArabiCCR-dataset.csv`;
- size: 305,354,149 bytes;
- SHA-256: `1064ede6fd2e9d697be258cfaff0594559599d0ca8ba39bc0100017f12a0f9ef`;
- rows: 12,806; schema fingerprint: `b8faee2d261232501bc28143765c21ca8f296496623818272d1f499ab1b599e2`;
- columns: `judgment_number`, `case_number`, `court_name`, `case_type`, `judgment_date`, `year`, `city`, `details_url`, `case_text`, `EVENTS`, `REASONING`, `RULING`;
- exact duplicate rows: 0; physical duplicate files: 0.

Acquisition method: authorized manual official download from the canonical Mendeley Data release followed by the policy-authorized local import command. The committed manifests contain the stable source/version identity, canonical-source metadata, repository-relative filename, size, and SHA-256; no user path or download timestamp is recorded.

Its sanitized privacy summary reports 31,314 findings across all 12,806 records, with 100 records selected for deterministic review. Detector counts are email 118, IBAN-like 4, and phone 31,192. Column counts are `EVENTS` 3,977, `REASONING` 1,159, `RULING` 510, `case_number` 3,721, `case_text` 5,420, `judgment_date` 12,806, and `judgment_number` 3,721. Confirmed PII and likely false-positive classifications are `not_reviewed`; the status is `pending_manual_review`. These findings remain unresolved signals; the claimed anonymisation is not treated as legal clearance.

### MOJ-derived statutory seed

The pinned Hugging Face revision `8b55ef5a666ad773c81086051813582fd14eb466` was acquired from `WafaaFraih/saudi-legal-moj` through the official Hub adapter. Files recorded in the lock are:

| File | Size | SHA-256 |
| --- | ---: | --- |
| `README.md` | 6,041 bytes | `3162fe8ddee605ab2a293239239031e7d80108bd7f051aa77f02cd9f78115520` |
| `data/train-00000-of-00001.parquet` | 1,350,919 bytes | `d1207f190afadb5baa24d3a8c0666f34f17b2d880b5319537e832abdefb75fae` |

The Parquet file contains 3,185 rows and columns `text`, `article_number`, `law_name`, `law_type`, and `source`, with schema fingerprint `599ba26e885ef25c5c87f43606185b6777b0e7219531355986ca3e078ed252c6`. The quality audit found 3,185 unique rows, 71 unique law names, 124 unique article labels, zero missing required fields, zero exact duplicate rows, 1,993 duplicate law/article keys, six extremely long records, and zero short, malformed/reversed-Arabic, suspicious-character, or likely merged-word/OCR records under the current deterministic heuristics. Date, status, amendment, and official-URL fields are unavailable in the seed.

The major quality warning is the 1,993 duplicate law/article keys: rows must not be treated as authoritative article units until a later reconstruction and reconciliation step. The seed is suitable as a raw parsing seed, but unsuitable as an authoritative or gold article corpus before reconciliation.

The seed is eligible for private Phase 3 parsing and quality inspection, not public display, training, or evaluation. Its uploader-declared CC BY 4.0 licence does not resolve MOJ source-text rights.

The MOJ seed privacy screen recorded 11 masked findings across 11 affected records, with a deterministic review sample of 11. Confirmed PII and likely-false-positive classifications are `not_reviewed`; the status is `pending_manual_review`.

### Initial v1 reconciliation candidates

The manually reviewable core subset is represented by 12 rows in `data/manifests/reconciliation/core-commercial-civil-v1.csv`: Commercial Courts, Civil Transactions, Bankruptcy, Arbitration, Enforcement, Evidence, Sharia Pleadings, Documentation, Legal Practice, Judicial Costs, Registered Mortgage, and Real Estate Finance. Corpus quality and domain coverage matter more than the raw count of 71 laws. This initial list is litigation-heavy and should be expanded or reprioritized in later curation to include core commercial instruments such as companies/business-formation and digital-commerce legislation when authoritative sources are identified. Official URLs, status, dates, amendments, article counts, samples, discrepancies, and eligibility remain blank or `pending_manual_reconciliation` until manually verified against the MOJ Legal Portal, the Bureau of Experts legislation portal, or the Saudi National Platform rules catalogue. No portal scraping was performed.

### Phase 2B rights and next-stage decisions

ALARB and ArabiCCR retain private local parsing/evaluation authorization with `legal_clearance=false`, `authorized_for_training=false`, and `authorized_for_public_display=false`. The MOJ-derived seed has `legal_clearance=false`, `authorized_for_local_parsing=true`, `authorized_for_evaluation=false`, `authorized_for_training=false`, and `authorized_for_public_display=false` because it is a local-research-only derived seed pending official reconciliation. An official MOJ article-level export remains the preferred replacement for any approved statutory corpus.

### Governance evidence and official reconciliation authorities

Saudi Copyright Law Article 4 is recorded as a rights-layer evidence item: Saudi laws, judicial judgments, administrative decisions, international agreements, official documents, and official translations are excluded from Saudi copyright protection, subject to provisions concerning circulation. This is not blanket permission to scrape or mirror a government portal. The audit keeps underlying official-text copyright status, circulation restrictions, portal terms, derived dataset/compilation licence, privacy, automated-access permission, and public-display permission as separate questions. Non-commercial status does not override website terms or privacy rules.

Manual reconciliation may use the Saudi Ministry of Justice Legal Portal, the Saudi Bureau of Experts legislation portal, and the Saudi National Platform rules catalogue as official authorities. No bulk scraping is authorized; only manually selected records and user-provided official URLs may be reviewed.
