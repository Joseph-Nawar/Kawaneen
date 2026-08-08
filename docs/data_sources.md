# Phase 1 source-selection report

This report is the source-selection decision record for Phase 1. It describes evidence and constraints; it does not grant permission or imply that legal search, ingestion, or RAG exists.

## Comparison

| Source | Content / size / format | Role | Licence layers | Original-source rights | Privacy / access | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| ALARB | 13,341 Saudi commercial cases; Parquet; Arabic; 12,012 train / 1,329 test | benchmark / evaluation | Dataset card: Apache-2.0; paper/code/original rights separate | Unknown | High; public download, PII not independently inspected | `evaluation_only` |
| ArabLegalEval | 27,032 rows; Parquet; `ArLegalBench`, `MCQs`, `QA` subsets | benchmark / evaluation | HF dataset and GitHub code releases; dataset and code licences unknown | Unknown | High; access and privacy unverified | `metadata_only` |
| 9,699-case corpus | Full decisions, human summaries; 7,759 train / 1,940 test; categories 6,727 / 2,035 / 937 | benchmark / evaluation | Paper; repository dataset licence missing | Unknown | High; public repository path, no rights inferred | `blocked_pending_review` |
| ALCD | 3,170 Saudi cases, 47 classes, summaries/keywords/categories; CSV | benchmark / evaluation | Paper CC BY is article-only; Kaggle dataset licence unresolved | Unknown | High; public Kaggle page, OCR/quality issues | `blocked_pending_review` |
| ArabiCCR | 12,806 Saudi commercial cases; current CSV/XLSX; Mendeley v3, published 26 May 2026 | primary-corpus candidate / local research | Mendeley CC BY 4.0; paper and original rights separate | Unknown | High; anonymisation claimed, not inspected | `local_research_only` |
| Saudi Legal MOJ Dataset | 3,185 Arabic article records from 71 Saudi laws/regulations; Parquet; 5 columns | primary corpus / private local parsing | Uploader dataset card declares CC BY 4.0; source-text rights separate | Unknown; stated MOJ provenance is not an official export | Medium; automated screen pending manual review | `local_research_only` |
| Saudi MOJ designated open data | Named item not selected; likely aggregate records | reference / candidate | National Open Data License, item-level review required | Depends on item | Medium; public catalogue, text-bearing item unconfirmed | `conditional` |
| Saudi MOJ general portal | Public HTML pages and laws; no fixed corpus | reference | No blanket portal licence | Unknown | Medium; public web | `metadata_only` |
| Saudi Board of Grievances open data | Open-data statistics: cases, judgments, hearings | reference | Usage guide requires attribution and no distortion | Does not establish full-text rights | Medium; public catalogue, aggregate not text corpus | `metadata_only` |
| Saudi Bureau of Experts | Official legislation portal | reference | Terms conflict with unapproved automation/redistribution | Unknown | Medium; public web, terms restrictive | `excluded` |
| Egyptian Court of Cassation | Subscriber legal-information site | reference / local research | May 2026 terms permit limited research copying; no public redistribution | Court controls access | High; gated, extraction restrictions | `local_research_only` |
| Official Egyptian legislation candidates | Government-domain candidates; exact portal not selected | reference | Not verified | Unknown | Medium; unknown access/licence | `metadata_only` |
| UAE Legislation | Official federal legislation platform | reference | Platform copyright/access licence not verified | Unknown | Medium; public web | `metadata_only` |

## Rights interpretation

Paper, code, dataset, and original-source rights are separate fields in the registry. A paper's CC BY licence covers the article, not automatically a dataset described by it. A repository's visibility does not grant dataset rights. A Mendeley or Hugging Face dataset licence does not resolve rights in underlying judgments, personal data, or public demonstration. Automated access, public display, model training, and public-demo permissions are separately recorded.

`source_role` describes intended research function: primary corpus, benchmark, training, transfer, or reference. `decision` describes legal/governance disposition. Thus `evaluation_only` and `local_research_only` are decisions, not roles.

## Groupings

- Publicly inspectable metadata only: all 13 records have canonical metadata pages, but no record is approved for public legal-text display.
- Local-only candidates: ArabiCCR, the MOJ-derived seed, and Egyptian Court of Cassation, subject to their conditions and manual review.
- Evaluation candidates: ALARB is evaluation-only pending source-rights and privacy review; the 9,699-case and ALCD datasets remain blocked, not evaluation-approved.
- Excluded: Saudi Bureau of Experts portal for automated corpus work under its current terms.
- No public-demo corpus: the audit found no legally adequate public-demo legal-text corpus. A separately licensed demo corpus must be sourced before a public demo.

## Recommended Phase 2 acquisition strategy

1. Do not use generic Saudi MOJ open data or portal scraping as the primary corpus. The acquired named MOJ-derived seed is a private local research input and raw parsing seed only; reconcile its 10–20-law commercial/civil subset manually against the MOJ Legal Portal, Bureau of Experts portal, and Saudi National Platform rules catalogue.
2. Proposed local research corpus: ArabiCCR version 3 (`10.17632/np538c95yy.3`) and the pinned MOJ-derived seed, only for controlled local parsing after privacy and rights review.
3. Proposed public-demo corpus: none of the audited case sources. Obtain a separate licensed, synthetic, author-cleared, or public-domain legal-text corpus with explicit public-display and demo permission.
4. Proposed evaluation corpora: ALARB for benchmark evaluation under its controlled decision. The MOJ-derived seed is not evaluation-approved until official reconciliation; ALCD and the 9,699-case corpus remain blocked until their dataset licences and source rights are verified.
5. Use official Saudi, Egyptian, and UAE portals as reference links until page-specific automation and quotation permissions are recorded.

## Exact unresolved actions

- Confirm ALARB original-record rights and PII mitigation; the registry records 12,012 train and 1,329 test.
- Obtain a dataset licence and confirm source rights, privacy, training, and display permissions for the 9,699-case repository/paper combination.
- Verify the ALCD Kaggle licence and Board of Grievances terms; do not rely on the article CC BY notice.
- Inspect ArabiCCR v3 files for residual identifiers and confirm MOJ provenance rights.
- Reconcile the MOJ-derived seed’s 12 core commercial/civil candidates against official MOJ URLs, status, dates, amendments, and article counts.
- Expand or reprioritize the litigation-heavy 12-law list to include core companies/business-formation and digital-commerce instruments when authoritative records are identified.
- Request an official machine-readable MOJ article-level export and explicit automated-access, processing, quotation, redistribution, and demo terms.
- Select a named text-bearing MOJ item, if one exists, and record its item-level terms.
- Confirm whether Board open-data catalogue items are aggregate only or include reusable text.
- Obtain written permission before using Bureau of Experts, Egyptian Court, Egyptian legislation, or UAE platform content for automated extraction or public display.
