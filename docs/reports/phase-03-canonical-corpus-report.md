# Phase 3 canonical corpus qualification report

Date: 2026-08-10

## Scope and qualification semantics

This phase constructed exact-text canonical local views and conservative statutory
reconstruction metadata. It did not normalize Arabic, chunk for retrieval, embed,
train, serve, publish, or modify Phase 2 raw files. Qualification is reported as
separate gates; a missing empirical benchmark is not a pass.

## Canonical corpus gate

| Source | Version | Raw records | Canonical documents | Units | Fragments | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ALARB | `e64bfdc867146294a65434c5ca16c2c4c5288ca2` | 13,341 | 13,341 | 53,364 | 0 | passed: complete accounting; official train/test preserved |
| ArabiCCR | `3` | 12,806 | 12,806 | 51,224 | 0 | passed: complete accounting; no modelling split created |
| `saudi-moj-derived` | `8b55ef5a666ad773c81086051813582fd14eb466` | 3,185 | 71 law documents | 3,185 | 3,185 | passed as private seed only; all rows retained as fragments |

Canonical output hashes, sizes, and repository-relative paths are in
`data/manifests/canonical/*.json`. The outputs are ignored local Parquet and are
not part of the staged snapshot. Phase 2 raw hashes were rechecked after the build
and were unchanged.

## Statutory structural gate — corrected parser rebuild

The original parser used substring matching and unrestricted digit extraction. The old
1,192 groups, 446 duplicate groups, and 2,439 duplicate-key excess rows are therefore
superseded. The corrected rebuild read the unchanged 3,185 raw fragments and produced
949 structural groups, 450 duplicate groups containing 2,686 rows, and 2,236
duplicate-key excess rows.
The rebuild parsed 3,174 labels with high confidence and left 11 non-article or
unknown labels unresolved. It identified 344 part-marked records, 25 ambiguous
continuation candidates, zero high-confidence continuation candidates, 425 genuine
conflict candidates, and 23 explicit fragment-series groups. No ambiguous group was
merged. Full distributions and per-law metrics are in the regenerated canonical
manifests.

The duplicate diagnostic now uses full structural-label keys and separates article
ordinal from part index. Its 25-group sample remains metadata-only and the no-merge
policy remains in force. The 3,185 raw rows remain individually provenance-linked;
the derived groups are a private parsing seed and are not authoritative article units.

The corrected deterministic 25-group sample contains 25 `genuine_conflict_candidate`
records under the regenerated structural key diagnostic; it has not received a new
text-level independent review. The prior 23 `malformed_source_grouping` and 2
`nested_subarticle_structure` classifications are historical and superseded by the
parser correction. No group was merged. The sample remains non-authority metadata;
future structural review may address explicit part markers only after independently
verified ordering evidence and additional tests.

The `groups` metric counts unique full structural-label groups, not authoritative
articles. For example, 95 Commercial Courts fragments produce 27 derived groups
because explicit part-labelled fragments share an article key while unresolved labels
remain row-specific. The private core-law structural audit records one row per
fragment and passed assertions that full ordinals such as 4/14/24 and 1/101/201 do
not share groups, that part numbers do not replace article ordinals, and that
unresolved labels do not merge by partial similarity.

## Authoritative statutory reconciliation gate

All 12 core candidates now have individually inspected Bureau of Experts detail URLs,
confirmed official titles, local seed article counts, and explicit `not_verified` or
`not_reviewed` values where official dates, amendment details, official article counts,
or risk-based article samples were not established. The two detail pages whose public
metadata exposed dates retain those dates; no unavailable field was guessed.

Every record is `present_partial` with `not_eligible_pending_reconciliation` status. This is
meaningful progress, not completion: manual first/early/middle/late and collision
sample review, official article-count comparison, and amendment verification remain.
The authorities are the Bureau of Experts legislation portal, Saudi National Platform
rules catalogue, and MOJ Legal Portal; no bulk scraping occurred.

The current one-by-one refresh encountered timeouts on all 12 individual official
BOE lookups. The supplied authoritative metadata draft was schema- and identity-
validated and its supported titles, URLs, statuses, dates, amendment notes, and
numbering evidence were incorporated into the sanitized CSV. It is metadata evidence
only, not completed local-vs-official text reconciliation; sample results and v1
eligibility remain unresolved. No law is v1-eligible.

| Instrument | Official record | Local groups | Sample comparison | v1 status |
| --- | --- | ---: | --- | --- |
| نظام المحاكم التجارية | Bureau of Experts URL retained | 27 | not_verified | not eligible |
| نظام المعاملات المدنية | Bureau of Experts URL retained | 28 | not_verified | not eligible |
| نظام الإفلاس | Bureau of Experts URL retained | 25 | not_verified | not eligible |
| نظام التحكيم | Bureau of Experts URL retained | 18 | not_verified | not eligible |
| نظام التنفيذ | Bureau of Experts URL retained | 23 | not_verified | not eligible |
| نظام الإثبات | Bureau of Experts URL retained | 28 | not_verified | not eligible |
| نظام المرافعات الشرعية | Bureau of Experts URL retained | 28 | not_verified | not eligible |
| نظام التوثيق | Bureau of Experts URL retained | 18 | not_verified | not eligible |
| نظام المحاماة | Bureau of Experts URL retained | 18 | not_verified | not eligible |
| نظام التكاليف القضائية | Bureau of Experts URL retained | 13 | not_verified | not eligible |
| نظام الرهن العقاري المسجل | Bureau of Experts URL retained | 11 | not_verified | not eligible |
| نظام التمويل العقاري | Bureau of Experts URL retained | 10 | not_verified | not eligible |

The CSV remains the authoritative sanitized reconciliation record, including exact
URLs, titles, dates, controlled unresolved fields, and manual actions.

The updated gap report classifies the 12 represented instruments as
`present_untrusted` and the remaining eight desired commercial/civil targets as
`missing_targeted_acquisition`. The current candidate list is litigation-heavy;
domain coverage matters more than the raw count of 71 titles, and later curation
should add companies/business formation and digital-commerce instruments when
authoritative records are identified.

## PDF/layout parser and OCR gates

Three authorized private PDFs were preflighted: 38 total pages, 23 pages with
embedded text, and 7 image-only pages. A deterministic stratified sample selected 30
real pages: 3 MOJ pages, 7 SAMA image-only pages, and 20 spread Umm Al-Qura pages,
including four likely complex-layout pages and one likely table/structured-box page.
Nine born-digital pages have 45 ignored controlled variants at approximately 200 DPI,
150 DPI, and deterministic rotation/contrast/blur settings. They are labelled
controlled variants, not historical scans.

Visual spot-checks of representative MOJ, SAMA, and Umm Al-Qura pages confirmed
born-digital Arabic articles, image-only stamped material, and table/multi-column
newspaper-style layout. Poppler was unavailable, so the private visual render used
the installed pypdfium2 renderer; no rendered images are staged.

Candidate gold regions were generated independently from pypdf/source-page reference
content, but zero pages are human-verified. The external directory now contains the
validated AI/source review. All 30 frozen page IDs and all three source-PDF hashes
match the benchmark manifest; the adjudication contains 102 regions, including 21
SAMA visual transcriptions. It is recorded as `independent_ai_visual_review` and
`externally_source_verified`, never as human verification.

Metric denominators are separated in `data/manifests/parsing_benchmark.json`: three
MOJ pages for `legal_structure`, seven SAMA pages for `legal_ocr`, ten Umm Al-Qura
regulatory pages for `legal_regulatory_layout`, three general complex-layout pages,
and seven non-legal complex-layout pages. The last two groups are excluded from legal
article-label and legal-structure denominators.

pypdf health probing passed. The previous RapidOCR 1.4.4 result was invalid because
its bundled detector/recognizer/classifier were Chinese PP-OCRv4 artifacts. The
replacement uses RapidOCR 3.9.2, ONNX Runtime 1.28.0, PP-OCRv5, and the explicit
Arabic mobile recognizer `arabic_PP-OCRv5_rec_mobile.onnx`. Its clean MOJ smoke test
produced 36 non-empty regions containing Arabic code points in 1.383 seconds; the
three model SHA-256 values are in the committed benchmark manifest. Model-weight
terms remain separate from the Apache-2.0 runtime and unresolved.

The previous Docling termination was not reproduced with an explicit OCR-off,
table-off layout pipeline and pypdfium2 backend. A one-page subprocess returned a
usable structured document on CPU in 5.760 seconds and auto/CPU in 5.705 seconds;
the sanitized records are in `data/manifests/parsing_diagnostics.json`. Auto selected
CPU, so no MPS conversion was exercised.

The final external-AI diagnostic covered all 30 pages and 102 regions with zero page
failures or timeouts. By scope: `legal_structure` recorded CER 0.718%, WER 1.667%,
exact and semantic article accuracy 100%, reading order 100%, and page preservation
100%; heading F1 was 0.333. `legal_ocr` recorded CER 9.535%, WER 27.511%, heading
F1 0.662, reading order 100%, and page preservation 100%. `legal_regulatory_layout`
recorded CER 18.688%, WER 37.980%, heading F1 0.350, semantic article accuracy
20.0%, reading order 71.667%, and page preservation 100%. General/non-legal scopes
were reported diagnostically only and excluded from legal article denominators. The
external Umm Al-Qura scope labels were corrected to the canonical manifest: p01–p03
are regulatory and p16/p18/p20 are general-layout pages.
The 45 controlled variants also completed under a 30-second watchdog: 18
clean/low-resolution and 27 degraded, all with non-empty Arabic output, averaging
2.427 seconds per variant; they are not scored without mapped gold.

The failures are source/route-specific: embedded-text legal recognition passes
text and article-number thresholds, while full-page OCR and complex-layout routes
fail the provisional CER/WER/heading gates. The heading result is additionally
limited by the current adapter's text-shape heuristic rather than typed Docling
layout blocks. An initial evaluation-script defect (reversed match tuple and
collapsed embedded lines) was corrected before these final metrics were recorded.

The implementation boundary and licensing review are documented in
`docs/parsing-and-ocr.md`: pypdf BSD-3-Clause, Docling MIT, RapidOCR runtime
Apache-2.0, and the exact PP-OCR weight release still requiring separate licence
verification. No weights are bundled or downloaded automatically, and PyMuPDF is
not a direct dependency.

## Gate summary

| Gate | Status | Reason |
| --- | --- | --- |
| canonical case corpus | passed for private local parsing/evaluation | complete accounting and stable provenance |
| statutory structural reconstruction | passed as conservative metadata | all collisions diagnosed; no ambiguous merge |
| authoritative statutory reconciliation | incomplete | 12 records are `present_partial`; manual samples/counts remain |
| PDF/layout parser qualification | not qualified | external diagnostic is integrated, but heading/layout gates fail and typed layout metrics remain limited |
| OCR qualification | not qualified | legal OCR CER/WER/heading gates fail despite zero timeouts and successful Arabic output |

Phase 3 is therefore not genuinely ready to close. Remaining actions are to
complete the 12-law independent statutory reconciliation, obtain project-owner
signoff for the priority benchmark pages if human verification is required, diagnose
the failing OCR/layout routes, verify exact model-weight terms, and keep all
training/public-display restrictions.

## Verification record

The remediation verification includes `uv sync --locked --dev --extra parsing`, source
registry validation, Phase 2 data verification, the corpus commands, PDF preflight,
CPU/auto Docling diagnostics, the Arabic smoke test, Ruff, Pyright, pytest,
pre-commit, `make check`, and both diff checks. The 23-page born-digital/complex
and seven-page image-only diagnostic batches completed; the 45 controlled variants
also completed with per-variant watchdogs. The current test suite passed 158 tests
with 85.43% branch-aware coverage.
No raw files, canonical Parquet, private benchmark material, OCR caches/models, or
credentials are staged. External-gold integration passed page-ID/hash validation and
is recorded as AI/source-verified only; `human_verified` remains false until human
review.

## Corrective requalification status

The statutory derived layer has been rebuilt and the parser regression suite covers
compound ordinals, hundreds, joined/spaced forms, part independence, suffixes, and
fail-closed malformed labels. ALARB remains 13,341 documents and ArabiCCR remains
12,806 documents with their prior provenance and restrictions. Phase 2 raw hashes
remain unchanged.

The external independent AI adjudication and its 21 SAMA transcriptions are now
validated and integrated as external diagnostic evidence. The corrected 60-sample
statutory handoff remains pending authoritative review. The PDF/layout and OCR gates
remain open because the final scoped metrics fail provisional quality thresholds and
the owner has not marked the evidence human-verified. Model-weight licence terms
also remain unresolved.

## Final consolidation audit — 2026-08-10

The exact external review filenames were validated in the external review directory;
the statutory adjudication JSON was not supplied. The regenerated private handoffs
are:

- statutory review: `artifacts/private/handoff/phase3-independent-review/phase3_statutory_review_bundle.jsonl` and `.md`;
- corrected 12-law fragment audit: `artifacts/private/handoff/phase3-independent-review/phase3_statutory_structural_audit.csv`;
- owner checklist: `artifacts/private/parsing_benchmark/project_owner_signoff_checklist.md`.

Validated external evidence hashes are: gold JSONL
`e1f8812b81afd550cbaefae351a4ed8ba089645ab1c900792c8d824deea4099c`, review report
`5ee4d79074e7878bc52b8fcd563ed5c052e8e3edea16816734a4c8ef1c43f69b`, and metadata
draft `592919c3710f77757cfbb03b573f2ac0c2f5c5d9dd87e49ed06758709350e883`.
The final statutory handoff is
`artifacts/private/handoff/phase3-independent-review/phase3_statutory_review_bundle.jsonl`
with SHA-256 `62b85245d0838f9d9976304699c54302e4816758124ddd4555625cfc128f7a67`,
12 laws, 60 samples, and revision
`8b55ef5a666ad773c81086051813582fd14eb466`.

The private structural audit covers 1,233 fragments across all 12 core laws and 247
derived groups. Its automated assertions passed: confidently parsed full ordinals do
not share groups; part 2/3 markers cannot replace article 60; and unresolved labels
remain non-merging. Across all 3,185 statutory fragments, 3,174 (99.655%) parse with
high confidence, 11 remain unresolved, and the corrected layer has 949 groups, 450
duplicate groups, 2,686 rows in duplicate groups, and 2,236 duplicate-key excess
rows. The former 1,192/446/2,439 statistics remain historical only.

All 45 controlled variants completed with a 30-second watchdog: 18 clean/low-
resolution and 27 deterministic degraded scans, all non-empty with Arabic output,
at a mean 2.427 seconds per variant. They remain unscored because mapped gold for
the variants was not supplied.
The three MOJ and seven SAMA legal pages remain the owner signoff priority; no
`human_verified` state has been set. Individual official lookups for all 12 BOE
records timed out, so titles/URLs already evidenced were retained and status, dates,
amendments, official counts, and sample comparisons remain unresolved.

This audit does not close Phase 3: canonical case construction and conservative
statutory structural reconstruction pass their local evidence gates, while
authoritative reconciliation, PDF qualification, and OCR qualification remain open.
