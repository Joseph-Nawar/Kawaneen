# Phase 1 Data-Source and Licensing Audit Report

## Scope and outcome

Phase 1 adds an offline source-governance layer and a metadata-only audit of 12 candidate legal sources. No full dataset, case text, court record, credential, or generated build artifact is included. No ingestion, parsing, OCR, NLP, retrieval, API, or model functionality is implemented.

## Sources inspected

ALARB; ArabLegalEval; the Saudi 9,699-case summarisation/classification candidate; ALCD 3,170-case candidate; ArabiCCR; Saudi Ministry of Justice designated open data; Saudi Ministry of Justice general portal content; Saudi Board of Grievances open-data candidate; Saudi Bureau of Experts legislation portal; Egyptian Court of Cassation; other official Egyptian legislation candidates; and UAE Legislation.

Each source has a separate evidence note under [`docs/source-audits/`](../source-audits/) and a corresponding row in [`data/manifests/source_registry.csv`](../../data/manifests/source_registry.csv).

## Evidence hierarchy

The audit prioritised source-specific licences and terms; official dataset cards and government open-data policies; repository licence files; official papers; then discovery leads. Paper evidence was never treated as dataset permission. Official authority and public availability were never treated as redistribution or automated-access permission.

## Final decisions

| Decision | Sources | Count |
| --- | --- | ---: |
| `approved` | None | 0 |
| `conditional` | Saudi Ministry of Justice designated open data | 1 |
| `evaluation_only` | ALARB | 1 |
| `local_research_only` | ArabiCCR; Egyptian Court of Cassation | 2 |
| `blocked_pending_review` | Saudi 9,699-case candidate; ALCD | 2 |
| `metadata_only` | ArabLegalEval; Saudi Ministry general portal; Saudi Board of Grievances open data; Egyptian legislation candidates; UAE Legislation | 5 |
| `excluded` | Saudi Bureau of Experts legislation portal | 1 |

The Saudi national open-data licence is explicit about sharing, reuse, creation, and adaptation with attribution and licence notices. It still does not substitute for item-level privacy and terms review, so the record remains conditional.

## Privacy findings

Case-derived benchmarks and court records are high privacy risk until source-specific mitigation is documented. Official aggregate open data is currently low risk in the registry but remains subject to item-level review. General legal portals are medium risk because content scope and personal-data exposure are not established. No source is marked public-demo-safe without resolved evidence.

## Proposed corpus and evaluation strategy

No audited source is an approved public-demo corpus. The exact proposed local research corpus is ArabiCCR Mendeley v3, only after inspecting claimed anonymisation and confirming original MOJ rights. The exact evaluation candidate is ALARB, restricted to evaluation pending its source-rights and PII review; ALCD and the 9,699-case corpus remain blocked. A named Saudi MOJ text-bearing open-data item is a conditional future candidate, but aggregate open data is not a legal-text corpus. A separately licensed demo corpus is required before public demonstration. Official legislation portals remain reference-only.

## Unresolved manual actions

- ALARB: confirm original Saudi court-record rights, inspect PII/privacy mitigation, and obtain public-display/demo clearance. Its Apache-2.0 dataset licence is already recorded.
- ArabiCCR v3: inspect the claimed anonymisation, confirm original Saudi MOJ rights, and resolve training, public-display, and public-demo permissions. Its CC BY 4.0 dataset licence is already recorded.
- Saudi 9,699-case corpus: obtain a dataset licence and clarify original-source, privacy, training, and display rights.
- ALCD: verify the Kaggle dataset licence and complete Board of Grievances source-rights and privacy review.
- Confirm the exact Saudi Ministry open-data item, attribution obligations, privacy classification, update policy, and item-level terms.
- Confirm whether any Saudi Board of Grievances catalogue item is text-bearing and separately licensed from the aggregate statistics policy.
- Obtain written permission or a machine-readable licence for Saudi Bureau of Experts content before copying or automation.
- Confirm terms, copyright, access conditions, and privacy safeguards for Egyptian Court of Cassation content.
- Confirm the canonical Egyptian legislation portal and its reuse policy.
- Obtain explicit reuse and automated-access terms for UAE Legislation.

## Files changed

The Phase 1 snapshot adds `src/kawaneen/sources/`, focused source tests, the source registry and manifest README, 12 evidence notes, governance documentation, ADR 0004, the Phase 1 phase page, this report, and the Phase 1 implementation plan. It modifies the CLI and Makefile only for source validation and summaries.

## Historical baseline verification (superseded)

The following table records the earlier Phase 0/initial Phase 1 baseline only. Its counts and results are superseded by the final verification section below.

| Command | Result |
| --- | --- |
| `uv sync --locked --dev` | Passed; resolved 30 packages and checked 28 packages |
| `uv run kawaneen sources validate` | Registry valid: 12 records |
| `uv run kawaneen sources summary` | Passed; 12 sources with 4 blocked, 1 conditional, 1 excluded, and 6 metadata-only |
| `uv run kawaneen sources summary --format json` | Passed; JSON contained source count, decisions, roles, jurisdictions, and privacy risks |
| `uv run ruff check .` | Passed; all checks passed |
| `uv run ruff format --check .` | Passed; 45 files already formatted |
| `uv run pyright` | Passed; 0 errors, 0 warnings, 0 informations |
| `uv run pytest` | Passed; 28 tests and 91.18% branch-aware coverage |
| `uv run pre-commit run --all-files` | Passed; all 8 hooks passed |
| `make sources-validate` | Passed; registry valid: 12 records |
| `make sources-summary` | Passed; same 12-source decision summary |
| `make check` | Passed; formatting, lint, Pyright, source validation, and tests |
| `git diff --check` | Passed with no whitespace errors |

No commit or push is part of Phase 1.

## Final verification

The final documentation-consistency state has 12 valid registry records. Decision counts are: approved 0; conditional 1; evaluation-only 1; local-research-only 2; blocked-pending-review 2; metadata-only 5; excluded 1.

| Command | Result |
| --- | --- |
| `uv run kawaneen sources validate` | Passed; 12 records valid |
| `uv run kawaneen sources summary` | Passed; final decision counts shown above |
| `uv run kawaneen sources summary --format json` | Passed; JSON includes final decisions, roles, jurisdictions, privacy, access statuses, and file formats |
| `uv run ruff format --check .` | Passed; 45 files already formatted |
| `uv run ruff check .` | Passed; all checks passed |
| `uv run pyright` | Passed; 0 errors, 0 warnings, 0 informations |
| `uv run pytest` | Passed; 36 tests, 89.20% branch-aware coverage |
| `uv run pre-commit run --all-files` | Passed; all hooks passed |
| `make check` | Passed; formatting, lint, Pyright, registry validation, and tests passed |
| `git diff --check` | Passed; no whitespace errors |

The historical 28-test baseline and intermediate 33-test result are retained only as audit history, not as current results.

## Corrective content audit

The corrective audit was performed on 2026-08-06 after the initial registry passed structural validation but contained incomplete technical fields and discovery URLs.

### Factual corrections and fixes

- Added the required publisher, original publisher, task, language, size, unit, format, content unit, citation, quality, personal-data, access, authentication, and attribution fields to the typed model and CSV.
- Replaced discovery URLs with canonical primary records: the ALARB Hugging Face card and paper; the 9,699 MDPI paper and exact GitHub repository; the ALCD Data in Brief article, Kaggle page, and Kaggle DOI; current ArabiCCR Mendeley v3 and paper; Board open-data policy/catalogue; and the exact May 2026 Court of Cassation terms page.
- Corrected ALARB to 13,341 Arabic Parquet records with 12,012 train and 1,329 test records; corrected ArabiCCR to v3, DOI `10.17632/np538c95yy.3`, CC BY 4.0, published 26 May 2026, current CSV/XLSX metadata, 12,806 final cases, and claimed anonymisation; corrected ArabLegalEval to 27,032 Parquet rows and its three subsets; corrected the 9,699 splits/category counts and unresolved licences.
- Replaced free-text jurisdiction labels with controlled values and separated dataset, paper, code, original-source, automated-access, public-display, model-training, and public-demo rights.
- Added canonical-URL validation rejecting search-result URLs and expanded decisions to include `evaluation_only` and `local_research_only`.
- Reviewed attribution obligations: only ALARB, ArabiCCR v3, Saudi MOJ designated open data, and Saudi Board of Grievances open data retain `yes`; records without explicit licence/terms attribution evidence now use `unknown`.

### Decision changes

Final counts are: approved 0; conditional 1; evaluation_only 1; local_research_only 2; blocked_pending_review 2; metadata_only 5; excluded 1. No record is approved for public demonstration or primary-corpus acquisition.

### Exit-criteria assessment

The governance and audit deliverables are met: the registry is typed, canonical, offline-validating, fail-closed, and documented; every source has an evidence note; and the strategy distinguishes primary, local, public-demo, reference, and evaluation paths. Legal adequacy for any dataset acquisition is not yet met because required manual rights and privacy confirmations remain unresolved. Phase 1 is ready for review, not a permission to begin Phase 2.

### Corrective verification record

| Command | Result |
| --- | --- |
| `uv sync --locked --dev` | Passed; resolved 30 packages and checked 28 packages |
| `uv run kawaneen sources validate` | Passed; 12 records valid |
| `uv run kawaneen sources summary` | Passed; decisions 2 blocked, 1 conditional, 1 evaluation-only, 1 excluded, 2 local-research-only, 5 metadata-only |
| `uv run kawaneen sources summary --format json` | Passed; JSON includes decisions, roles, jurisdictions, privacy, access statuses, and file formats |
| `uv run ruff format --check .` | Passed; 45 files already formatted |
| `uv run ruff check .` | Passed; all checks passed |
| `uv run pyright` | Passed; 0 errors, 0 warnings, 0 informations |
| `uv run pytest` | Passed; 36 tests, 89.20% branch-aware coverage |
| `uv run pre-commit run --all-files` | Passed; all hooks passed |
| `make sources-validate` | Passed with the installed `uv` directory on PATH; 12 records valid |
| `make sources-summary` | Passed with the installed `uv` directory on PATH; corrected summary shown above |
| `make check` | Passed with the installed `uv` directory on PATH; formatting, lint, typecheck, registry validation, and tests passed |
| `git diff --check` | Passed; no whitespace errors |

The first Make attempt in the audit shell failed because `uv` was installed but not on PATH; no project defect was found, and all Make targets passed when the installed `uv` directory was added to PATH. No external dataset, case text, credentials, or build artefact was staged. No commit or push was performed.

## Final Factual Consistency Review

- ArabiCCR is now source ID `arabiccr`, with Mendeley version 3, DOI `10.17632/np538c95yy.3`, canonical release URL, publication date 26 May 2026, CC BY 4.0, and current CSV/XLSX metadata. Its `local_research_only` decision and unresolved original-source, privacy, training, display, and demo rights are unchanged.
- ArabLegalEval now distinguishes the Hugging Face dataset release from the GitHub code release and records 27,032 Parquet rows across `ArLegalBench`, `MCQs`, and `QA`; dataset and code licences remain unknown and the decision remains `metadata_only`.
- ALARB now records 12,012 train and 1,329 test records; split-count review was removed from manual actions and `evaluation_only` remains unchanged.
- The 9,699-case corpus now records 7,759 train, 1,940 test, and category counts of Administrative 6,727, Commercial 2,035, and Criminal 937. It remains `blocked_pending_review` because licensing, source, privacy, training, and display rights are unresolved.
- Attribution was made evidence-based. A `yes` value requires licence or terms evidence; missing-licence records cannot assert a licence-derived attribution obligation without separate terms evidence. Regression tests cover both rejection and separately evidenced acceptance.
