# Phase 3 parsing qualification report

Date: 2026-08-11

## Decision

The benchmark is now methodologically valid and Phase 3 can close. No route
qualified for automatic v1 ingestion. Each measured route is explicitly
`manual_review_or_abstain`; this is a truthful support result, not a request for
another tuning cycle. No commit or push was performed.

## Anchored Gold v2

The frozen 30 selected pages and 102 externally reviewed region strings were
preserved. The private artifact is
`artifacts/private/parsing_benchmark/anchored_gold_v2.jsonl`; it uses canonical
top-left PDF-point coordinates and records the source PDF hash, page dimensions,
region type, semantic article number where applicable, anchor method, and review
provenance. The deterministic visual audit is
`artifacts/private/parsing_benchmark/contact_sheet_v2.png`.

| Source | Pages | Regions | Anchor method |
| --- | ---: | ---: | --- |
| MOJ | 3 | 11 | 11 PDFium text-geometry anchors; 1 required independent visual disambiguation |
| SAMA | 7 | 21 | 21 independent visual anchors from the private canvas annotation records |
| Umm Al-Qura | 20 | 70 | 66 PDFium text-geometry anchors; 3 source-text visual disambiguations; 1 independent source visual anchor |
| **Total** | **30** | **102** | **76 PDFium text geometry, 4 PDFium visual disambiguations, 1 independent source visual anchor, 21 independent SAMA visual anchors** |

The born-digital process located the reviewed span in the PDF text layer,
unioned PDFium character rectangles, and converted PDF bottom-left coordinates
to the canonical top-left system. Locator normalization was limited to matching
whitespace, presentation forms, tatweel, and extraction-spacing differences; the
stored gold text is the exact reviewed string. Ambiguous matches fail closed.

Visual/manual records were required for:

- all 21 SAMA regions (`sama_bankruptcy_obligations__p01` through `__p07`),
  recorded as `independent_ai_visual_anchor`, `human_verified=false`;
- MOJ p01 region 01, Umm Al-Qura p01 region 02, p20 region 03, and p26 region
  03 for independent visual disambiguation of repeated or extraction-damaged
  source text;
- Umm Al-Qura p24 region 02, where the independent source visual box was used
  because the PDFium footer geometry did not cover the rendered crop.

## Integrity and governance audit

Passed before scoring:

- exactly 30 selected pages and 102 regions;
- 102 unique region IDs, non-empty exact gold strings, and 102 non-empty
  in-bounds boxes;
- all three source PDF hashes unchanged;
- deterministic top-left coordinate conversion and page dimensions;
- no parser/OCR-derived gold text or final geometry;
- no ambiguous born-digital anchor accepted;
- every region box rendered in the private contact sheet;
- ALARB and ArabiCCR canonical manifests had no working-tree changes during this
  task;
- 3,185 statutory fragments remain accounted for;
- statutory reconciliation remains `completed_partial_not_eligible`;
- all six Phase 2 raw files match `data/manifests/raw_file_manifest.csv`;
- no private gold, source PDF, model, or raw artifact is staged.

The read-only audit output is
`artifacts/private/parsing_benchmark/phase3_final_audit.json`.

## Frozen development/holdout split

The split is versioned in `data/evaluation/phase3_split.json` and was frozen
before route results:

| Stratum | Development | Holdout |
| --- | ---: | ---: |
| MOJ legal structure | 2 pages (p01–p02) | 1 page (p03) |
| SAMA legal OCR | 4 pages (p01–p04) | 3 pages (p05–p07) |
| Umm Al-Qura regulatory layout | 7 pages (p01–p06,p08) | 3 pages (p10,p12,p14) |
| Other frozen diagnostics | 9 pages | not used for route selection |
| **Total** | **21** | **9** |

Only development pages selected configurations. The holdout pages were each
evaluated exactly once; immutable per-route ledgers record that count.

## Selected configurations

- MOJ legal structure: `docling_layout` (the only bounded route candidate).
- SAMA legal OCR: `full_page_200`, selected from the existing 200/250/300 DPI,
  grayscale/contrast, threshold, and conservative-orientation candidates.
- Umm Al-Qura regulatory layout: `docling_layout`, selected against the existing
  `pypdfium2_text_lines` candidate.

The evaluator converts Docling, PDFium, and OCR boxes into the same canonical
top-left coordinates. Correspondence is geometry-only: predicted blocks are
selected by documented box-overlap/center rules, ordered by predicted reading
order with geometry tie-breaking, and only those blocks are concatenated for a
gold region. Headings and article labels are typed/geometry matched, and reading
order is pairwise over anchored region IDs.

## Results

The unchanged provisional Phase 3 gates were: CER ≤ 0.05, WER ≤ 0.15,
heading F1 ≥ 0.80, exact and semantic article accuracy ≥ 0.95, pairwise reading
order ≥ 0.95, page-reference preservation = 1.00, critical article errors = 0,
and zero failures/timeouts. Runtime is seconds/page.

### Development metrics for the selected configuration

| Route | CER | WER | Heading P / R / F1 | Exact article | Semantic article | Order | Page refs | Critical errors | Runtime/page | Failures/timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MOJ | 0.4200 | 0.4773 | 0.0833 / 1.0000 / 0.1429 | 0.5000 | 0.5000 | 1.0000 | 1.0000 | 1 | 3.0528 | 0 / 0 |
| SAMA | 0.4373 | 0.6096 | 0.2500 / 0.0833 / 0.1250 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0 | 0.9712 | 0 / 0 |
| Umm Al-Qura | 0.6472 | 0.8448 | 0.0962 / 1.0000 / 0.1658 | 0.1429 | 0.0000 | 0.5238 | 1.0000 | 3 | 0.8749 | 0 / 0 |

### Final holdout metrics versus gates

| Route | CER (≤.05) | WER (≤.15) | Heading P / R / F1 (F1 ≥.80) | Exact article (≥.95) | Semantic article (≥.95) | Order (≥.95) | Page refs (=1) | Critical errors (=0) | Runtime/page | Failures/timeouts | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MOJ / Docling | 0.7380 **FAIL** | 1.0000 **FAIL** | 0.0000 / 1.0000 / 0.0000 **FAIL** | 1.0000 pass | 1.0000 pass | 1.0000 pass | 1.0000 pass | 0 pass | 0.6630 | 0 / 0 | `manual_review_or_abstain` |
| SAMA / RapidOCR 200 DPI | 0.4498 **FAIL** | 0.6946 **FAIL** | 0.5000 / 0.5000 / 0.5000 **FAIL** | 0.0000 **FAIL** | 0.0000 **FAIL** | 1.0000 pass | 1.0000 pass | 0 pass | 1.1389 | 0 / 0 | `manual_review_or_abstain` |
| Umm Al-Qura / Docling | 1.0708 **FAIL** | 1.2950 **FAIL** | 0.0175 / 0.6667 / 0.0333 **FAIL** | 0.3333 **FAIL** | 0.0000 **FAIL** | 0.4444 **FAIL** | 1.0000 pass | 2 **FAIL** | 0.9597 | 0 / 0 | `manual_review_or_abstain` |

The complete per-page and candidate ledger is retained privately at
`artifacts/private/parsing_benchmark/phase3_qualification_results.json`.

## Support matrix and deployment-only issue

| Route | Automatic v1 ingestion | v1 support |
| --- | --- | --- |
| MOJ legal structure | No | `manual_review_or_abstain` |
| SAMA legal OCR | No | `manual_review_or_abstain` |
| Umm Al-Qura regulatory layout | No | `manual_review_or_abstain` |

There is no remaining benchmark-method blocker. Before deployment, the Arabic
PP-OCRv5 model-weight terms and the optional Docling dependency/licence closure
must be reviewed; the model files remain local/ignored and are not shipped.

## Verification and gate table

The focused anchoring, geometry, isolation, split-freeze, and one-shot holdout
tests pass. The final run also includes the repository test suite, Ruff, Pyright,
pre-commit, `make check`, deterministic artifact comparisons, the contact-sheet
audit, raw-hash verification, and staged-private-artifact checks.

| Gate | Result |
| --- | --- |
| Frozen 30 pages / 102 regions preserved | PASS |
| Independent, valid anchored gold | PASS |
| Geometry-based correspondence and metrics | PASS |
| Ambiguity and invalid-box fail-closed tests | PASS |
| Frozen stratified dev/holdout split | PASS |
| Holdout evaluated once per route | PASS |
| Route quality thresholds | No automatic route passes; all correctly classified manual/abstain |
| Canonical/statutory/Phase 2 preservation | PASS |
| Private artifacts unstaged; no commit/push | PASS |
| Phase 3 closure | **PASS — close and merge-ready; no commit or merge performed** |
