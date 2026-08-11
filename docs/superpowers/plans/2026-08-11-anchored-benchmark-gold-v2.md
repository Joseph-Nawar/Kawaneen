# Anchored Benchmark Gold v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate independent anchored gold for the frozen 30 pages/102 regions, repair geometry-based scoring, freeze a dev/holdout qualification run, and report truthful Phase 3 support.

**Architecture:** Add focused private benchmark modules for anchoring/validation, geometry evaluation, qualification orchestration, and audit reporting. Keep the existing public parser routes and canonical/statutory paths unchanged; generated gold, renders, predictions, and reports remain ignored private artifacts.

**Tech Stack:** Python 3.12 (`>=3.11,<3.13`), pypdfium2/PDFium text geometry, existing Docling/RapidOCR adapters, Pydantic, pytest, Ruff, Pyright, pre-commit, uv.

## Global Constraints

- Preserve the frozen 30-page / 102-region externally reviewed gold and source PDF hashes.
- Do not modify the canonical case corpus, statutory parser/reconciliation, Phase 2 data, or begin Phase 4.
- Store no raw source material, credentials, models, OCR output, interim data, processed data, or private gold in version control.
- Use one normalized canonical top-left coordinate system internally.
- SAMA gold geometry must be independent of evaluated RapidOCR/Docling predictions.
- Select configurations only from development pages; evaluate each holdout page exactly once after freezing split/configuration.
- Do not change existing thresholds, gold, pages, split, or route candidates to obtain a pass.
- Do not commit or push.

---

### Task 1: Add red tests for coordinate and gold-integrity primitives

**Files:**
- Modify: `tests/test_parsing_metrics.py`
- Create: `tests/test_parsing_anchored_gold.py`

**Interfaces:**
- Tests target `pdf_bottom_left_to_top_left`, `scale_box`, `union_boxes`, `find_unique_text_anchor`, `validate_anchored_gold`, and `assign_blocks_to_gold_region`.

- [ ] **Step 1: Write failing tests** for bottom-left conversion, scaling, multiline union, ambiguous text failure, page/hash mismatch, invalid/out-of-bounds boxes, and unrelated-region isolation.
- [ ] **Step 2: Run the focused tests** with `uv run pytest tests/test_parsing_anchored_gold.py tests/test_parsing_metrics.py -q`; confirm failures are missing-function/behavior failures.

### Task 2: Implement anchored gold construction and fail-closed validation

**Files:**
- Create: `src/kawaneen/parsing/anchored_gold.py`
- Modify: `.gitignore`
- Test: `tests/test_parsing_anchored_gold.py`

**Interfaces:**
- `pdf_bottom_left_to_top_left(box, page_height) -> Box`
- `scale_box(box, source_width, source_height, target_width, target_height) -> Box`
- `union_boxes(boxes) -> Box`
- `find_unique_text_anchor(text, verified_text, page_width, page_height) -> Box`
- `validate_anchored_gold(records, expected_page_count=30, expected_region_count=102, source_hashes=None) -> GoldValidation`
- `build_anchored_gold(selection_path, external_gold_path, source_dir, output_path, audit_dir) -> GoldValidation`

- [ ] **Step 1: Implement typed record loading and canonical top-left geometry helpers.** Reject malformed boxes and preserve exact stored text.
- [ ] **Step 2: Implement PDFium text-span matching.** Normalize only the locator stream for whitespace/presentation-form/obvious extraction spacing; store the independently reviewed original text. Return no anchor for zero matches and raise an ambiguity error for multiple matches.
- [ ] **Step 3: Implement independent SAMA annotation loading.** Read only a private annotation file or fail closed with a concise instruction to run the annotation utility; never inspect predictions.
- [ ] **Step 4: Implement full validation and private JSONL emission.** Enforce counts, IDs, hashes, pages, dimensions, origin/system, anchoring provenance, nonempty bounds, and independence markers.
- [ ] **Step 5: Add ignored private output paths and run focused tests.**

### Task 3: Add private SAMA annotation utility and visual audit rendering

**Files:**
- Create: `tools/annotate_sama_gold.py`
- Create: `tools/render_benchmark_audit.py`
- Test: `tests/test_parsing_anchored_gold.py`

**Interfaces:**
- `annotate_sama_gold.py --pages-dir ... --gold ... --output ...` writes page/region boxes without reading prediction artifacts.
- `render_benchmark_audit.py --gold ... --pages-dir ... --output ...` renders deterministic overlays/contact sheet.

- [ ] **Step 1: Add a minimal annotation test fixture** proving the utility schema round-trips boxes and retains the three existing strings per SAMA page.
- [ ] **Step 2: Implement a dependency-light annotation UI** using deterministic page renders and editable rectangle input; record `independent_ai_visual_anchor`, `human_verified=false`, method/version, and page render hash.
- [ ] **Step 3: Implement contact-sheet rendering** with all 102 labeled boxes and deterministic ordering; fail if any region is missing.
- [ ] **Step 4: Generate/validate the private anchored gold.** Use PDFium for MOJ/Umm Al-Qura and the independent SAMA annotation records; if automatic visual placement is not reliable, do not synthesize it from OCR/parser output.

### Task 4: Replace evaluator correspondence with geometry-based region scoring

**Files:**
- Modify: `src/kawaneen/parsing/benchmark.py`
- Modify: `src/kawaneen/parsing/models.py`
- Modify: `src/kawaneen/parsing/docling_backend.py`
- Test: `tests/test_parsing_metrics.py`

**Interfaces:**
- `canonicalize_prediction_box(box, origin, page_height) -> Box`
- `assign_blocks_to_gold_region(gold_box, blocks, overlap_threshold=0.20, center_rule=True) -> tuple[int, ...]`
- `calculate_page_region_metrics(gold_regions, predicted_blocks) -> PageMetrics`

- [ ] **Step 1: Add tests** for spatial assignment, center/overlap rules, block ordering, typed heading/article metrics, page-reference preservation, critical article errors, and unrelated same-page regions.
- [ ] **Step 2: Convert Docling PDFium coordinates** to the canonical top-left system at the parser boundary while preserving provenance.
- [ ] **Step 3: Implement deterministic spatial assignment** restricted by page and region type; concatenate only assigned blocks in predicted reading order, then geometric tie-break order.
- [ ] **Step 4: Implement route/page aggregation** for CER, WER, heading P/R/F1, exact/semantic article accuracy, pairwise order, page references, critical article errors, runtime/page, failures, and timeouts.
- [ ] **Step 5: Update existing region tests and run focused metrics tests.**

### Task 5: Freeze split/configuration and enforce one-time holdout evaluation

**Files:**
- Create: `src/kawaneen/parsing/qualification.py`
- Create: `tests/test_parsing_qualification.py`
- Create: `data/evaluation/phase3_split.json`
- Create: `data/evaluation/phase3_qualification_report.json`

**Interfaces:**
- `create_frozen_split(selection_manifest, output_path) -> SplitManifest`
- `select_development_configuration(route, candidates, dev_pages) -> ConfigurationSelection`
- `evaluate_holdout_once(route, config, split, evaluator, ledger) -> HoldoutResult`
- `build_support_matrix(results, thresholds) -> SupportMatrix`

- [ ] **Step 1: Write tests** proving deterministic source-stratified split, candidate selection from development only, frozen configuration, and second holdout invocation failure.
- [ ] **Step 2: Implement split generation** across MOJ, SAMA, and Umm Al-Qura legal scopes with stable hash/order and no result-dependent page changes.
- [ ] **Step 3: Implement bounded development selection** for existing SAMA DPI/preprocessing/full-page-vs-layout candidates and Umm Al-Qura pypdfium2-vs-Docling/layout candidates; write a freeze ledger before holdout.
- [ ] **Step 4: Implement one-shot holdout evaluation** that refuses repeats and records failures/timeouts without silently retrying a holdout page.
- [ ] **Step 5: Run development selection, freeze the ledger, and evaluate each holdout exactly once.**

### Task 6: Produce final private audit/report and preserve governance state

**Files:**
- Create: `src/kawaneen/parsing/audit.py`
- Modify: `src/kawaneen/parsing/review.py`
- Modify: `docs/parsing-and-ocr.md`
- Modify: `data/manifests/parsing_benchmark.json`
- Create: `docs/reports/phase-03-parsing-qualification-report.md`
- Test: `tests/test_parsing_review.py`

**Interfaces:**
- `audit_phase3_state(...) -> AuditResult`
- `write_qualification_report(...) -> Path`

- [ ] **Step 1: Add tests** for unchanged ALARB/ArabiCCR manifests, 3,185 fragments, statutory status, raw hash stability, gold independence, geometry-only evaluation, and unstaged private artifacts.
- [ ] **Step 2: Implement audit checks** and report `manual_review_or_abstain` for failed optional routes without another tuning cycle.
- [ ] **Step 3: Emit final holdout metrics against unchanged thresholds**, with separate development metrics, selected config, split, anchoring counts/manual regions, critical errors, runtime, failures/timeouts, and gate table.
- [ ] **Step 4: Update Phase 3 documentation/manifest status only with measured results; do not alter statutory or corpus outputs.**
- [ ] **Step 5: Run deterministic audit and confirm no private gold/PDF/model artifacts are staged.**

### Task 7: Run the complete verification suite and handoff

**Files:**
- No production files; inspect all diffs and generated private artifacts.

- [ ] **Step 1: Run focused tests, then full `uv run pytest`.**
- [ ] **Step 2: Run `uv run ruff check .`, `uv run pyright`, `uv run pre-commit run --all-files`, and `make check`.**
- [ ] **Step 3: Run deterministic gold/evaluation/audit commands twice and compare hashes/results.**
- [ ] **Step 4: Run git diff, git diff --cached, status, and private-artifact staging checks; do not commit or push.**
- [ ] **Step 5: Return the requested counts, manual anchoring list, integrity audit, split/configs, holdout metrics vs thresholds, support matrix, tests/coverage, deployment issue, gate table, and closure decision.
