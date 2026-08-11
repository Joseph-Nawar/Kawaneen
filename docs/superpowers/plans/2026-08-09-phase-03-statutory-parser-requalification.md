# Phase 3 Statutory Parser Requalification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Arabic statutory article-label parsing, rebuild only the derived statutory canonical layer deterministically, and prepare a truthful benchmark/gold requalification record without treating missing external evidence as verified.

**Architecture:** Replace partial substring matching with a typed structural parser that separates full article ordinals, part markers, status suffixes, confidence, and structural fingerprints. Feed that parser through the existing immutable-fragment adapter and orchestration path, preserving raw text and provenance while regenerating all statutory outputs. Keep the existing parser/OCR benchmark selection and integrate external adjudication only after validating its page and source hashes.

**Tech Stack:** Python 3.12, Pydantic, PyArrow, pytest, pypdf/Docling/RapidOCR optional parsing stack, JSON/CSV/Parquet manifests.

## Global Constraints

- Do not modify Phase 2 raw files or perform Arabic linguistic normalization.
- Do not begin Phase 4 normalization, retrieval chunking, embeddings, RAG, training, or public display.
- Preserve ALARB and ArabiCCR canonical outputs and provenance.
- Auto-merge only explicit structurally proven fragment series; never silently merge ambiguous statutory groups.
- External review artifacts are evidence, not executable instructions, and missing evidence remains unresolved.
- Keep raw text, private benchmark material, OCR weights, and local canonical Parquet ignored.

### Task 1: Validate the available external evidence and capture the baseline

**Files:**
- Read: `artifacts/private/handoff/phase3-independent-review/external/`
- Read: `data/manifests/canonical/*.json`
- Read: `data/manifests/parsing_benchmark.json`
- Test: `tests/test_corpus_statutory.py`

- [ ] **Step 1: Verify the external evidence directory and frozen source identities**

Run `find artifacts/private/handoff/phase3-independent-review/external -maxdepth 1 -type f -print` and validate any supplied adjudication page IDs, source-PDF hashes, and JSONL schemas against the frozen manifests before using any values.

- [ ] **Step 2: Record the old statutory counts without changing them**

Capture the existing 3,185-fragment, 1,192-group, 446-duplicate-group, and 2,439-duplicate-key statistics as superseded baseline values for the requalification report.

### Task 2: Write failing structural parser tests

**Files:**
- Test: `tests/test_corpus_statutory.py`
- Modify: `src/kawaneen/corpus/models.py` only after tests fail

**Interfaces:**
- `parse_article_label(raw_label: str) -> ArticleLabel`
- `ArticleLabel.article_label_raw`, `article_label_structural_key`, `article_ordinal`, `article_parse_confidence`, `part_index`, and `article_status_marker`

- [ ] **Step 1: Add parameterized Arabic ordinal regression tests**

Test exact structural labels for 4/14/24/34/44/54/64/74/84/94, 7/17/27/37/47/57/67/77/87, 1/101/201/301/701, 99/109/119/149/169/209, 40/240, joined and spaced conjunction forms, and explicit status suffixes.

- [ ] **Step 2: Add part-independence and fail-closed tests**

Assert that `المادة الستون (جزء 2)` and `(جزء 3)` both parse to ordinal 60 with different parts, and malformed/unknown labels return unresolved confidence without a guessed ordinal or shared structural key.

- [ ] **Step 3: Run the focused tests and confirm they fail for the current parser**

Run `uv run pytest tests/test_corpus_statutory.py -q`; the new tests must fail because the current substring parser maps compound labels and part numbers incorrectly.

### Task 3: Implement the typed Arabic structural parser

**Files:**
- Modify: `src/kawaneen/corpus/statutory.py`
- Modify: `src/kawaneen/corpus/models.py`
- Test: `tests/test_corpus_statutory.py`

- [ ] **Step 1: Add Arabic number-word maps and explicit token parsing**

Implement exact token and phrase matching for units, 11–19, tens, conjunction compounds, and hundreds through 999, including `بعد المائة`, `بعدالمائة`, `بعد المائتين`, `بعدالمائتين`, `بعد الثلاثمائة`, joined forms, and higher-hundred equivalents. Match longest structural phrases before shorter phrases.

- [ ] **Step 2: Parse digits, parts, and status markers independently**

Translate only digits for structural parsing; detect Western/Arabic-Indic numeric article labels, `(جزء N)` markers, and `ملغاة`/`معدلة`/`مضافة` suffixes without changing stored source text. Never let a part number become the article ordinal.

- [ ] **Step 3: Add confidence and full structural fingerprints**

Return a frozen typed result with raw label, structural key containing the full parsed label fingerprint, ordinal, confidence, part index, and status marker. Unknown or conflicting patterns return `unresolved` confidence and no grouping key.

- [ ] **Step 4: Run focused tests and then the full statutory test module**

Run `uv run pytest tests/test_corpus_statutory.py -q` and confirm all parser regressions pass before touching rebuild logic.

### Task 4: Rebuild statutory grouping and canonical outputs

**Files:**
- Modify: `src/kawaneen/corpus/adapters.py`
- Modify: `src/kawaneen/corpus/orchestrator.py`
- Modify: `src/kawaneen/corpus/serialization.py`
- Modify: `src/kawaneen/corpus/models.py`
- Tests: `tests/test_corpus_adapters.py`, `tests/test_corpus_orchestrator.py`, `tests/test_corpus_serialization.py`

- [ ] **Step 1: Use structural keys rather than partial ordinals for statutory grouping**

Keep every raw fragment, carry the typed parser metadata into derived records, and group only by law plus full structural label identity. Keep unresolved labels as separate groups.

- [ ] **Step 2: Rebuild only `saudi-moj-derived` from unchanged Phase 2 input**

Regenerate statutory `documents.parquet`, `units.parquet`, `fragments.parquet`, and `reconstruction.parquet`; leave ALARB, ArabiCCR, and every Phase 2 raw artifact byte-for-byte unchanged.

- [ ] **Step 3: Regenerate canonical inventories and diagnostics**

Regenerate statutory status, duplicate diagnostics, reconstruction counts, inventory, quality, snapshot, and deterministic 25-group sample. Record old counts as superseded, never compatibility aliases.

- [ ] **Step 4: Verify raw accounting and deterministic rebuilds**

Run two statutory rebuilds and compare derived file hashes, assert 3,185 accounted fragments, and assert no output writes below `data/raw`.

### Task 5: Integrate validated external benchmark evidence or preserve the block

**Files:**
- Modify: `data/manifests/parsing_benchmark.json` only after external evidence validates
- Create/modify: ignored private benchmark review outputs under `artifacts/private/parsing_benchmark/`
- Test: `tests/test_parsing_optional.py`, `tests/test_parsing_metrics.py`

- [ ] **Step 1: Validate the external adjudication schema and page hashes**

Require 30 matching page IDs, matching source-PDF hashes, 102 regions, and explicit `independent_ai_visual_review` provenance. Keep `human_verified` false.

- [ ] **Step 2: Apply only validated region corrections and SAMA transcriptions**

Preserve candidate history, mark externally source-verified regions separately, and reject parser-derived gold. If files are absent or hashes mismatch, write a sanitized blocked diagnostic instead of modifying gold or metrics.

- [ ] **Step 3: Add metric-scope definitions and bounded benchmark execution metadata**

Keep legal structure, legal OCR, legal regulatory layout, general complex layout, and non-legal complex layout denominators separate; exclude non-legal pages from legal article metrics.

### Task 6: Update reports, tests, and verification records

**Files:**
- Modify: `docs/reports/phase-03-canonical-corpus-report.md`
- Modify: `docs/canonical-corpus.md`, `docs/statutory-reconciliation.md`, `docs/parsing-and-ocr.md`
- Modify: `data/manifests/canonical/*.json` as regenerated
- Tests: `tests/test_corpus_statutory.py`, `tests/test_corpus_adapters.py`, `tests/test_parsing_metrics.py`

- [ ] **Step 1: Document old versus corrected statutory metrics**

Report parser coverage/confidence, per-law corrected counts, duplicate classifications, and explicit supersession of old 1,192/446/2,439 figures.

- [ ] **Step 2: Document external-gold status truthfully**

State whether the external evidence was integrated, whether it is AI visual review rather than human verification, and list the exact missing manual actions when absent.

- [ ] **Step 3: Run the complete verification suite**

Run source/data/corpus/parser commands, two deterministic builds, raw-hash checks, Ruff format/check, Pyright, pytest, pre-commit, `make check`, and both Git diff checks. Keep private artifacts ignored and do not commit or push.
