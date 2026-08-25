# Phase 13 Product Quality, History, and Visual QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the targeted Phase 13 product gaps, rebase the Phase 13-only history onto current `main`, complete real browser QA and synthetic screenshots when tooling permits, and leave PR #9 clean and unmerged.

**Architecture:** Keep the existing Streamlit pages and typed Phase 12 `/v1` client. Add small pure presentation helpers for returned-result filtering, quote inspection, extraction summaries, pagination state, latency grouping, and evaluation comparison data; pages consume those helpers without importing serving or model runtime code.

**Tech Stack:** Python 3.12, Streamlit, Pydantic contracts, pytest/AppTest, Ruff, Pyright, `agent-browser` or the already-available browser surface, GitHub CLI.

**Spec:** `docs/superpowers/specs/2026-08-25-phase13-product-quality-followup.md`

## Global Constraints

- Streamlit consumes only the Phase 12 `/v1` HTTP API.
- Auto mode never silently masquerades demo data as live.
- Demo fixtures and screenshots are synthetic only.
- No unsupported metadata filters or fake confidence scores.
- Hybrid extraction remains visibly labelled `PHASE11_HYBRID_EXPERIMENTAL_LIMITED`.
- Evaluation metrics come only from sanitized, provenance-checked tracked sources.
- Do not access HOLDOUT, rerun evaluations, tune models, or alter frozen Phase 7–12 results.
- Do not merge PR #9.

---

### Task 1: Establish the ancestry and write failing product tests

**Files:**
- Inspect: `src/kawaneen/ui/pages/*.py`, `src/kawaneen/ui/formatting.py`, `src/kawaneen/ui/evaluation.py`, `tests/test_ui_*.py`
- Modify: `tests/test_ui_formatting.py`, `tests/test_ui_evaluation.py`, `tests/test_ui_state.py`, `tests/test_ui_app.py`

**Interfaces:**
- Tests define pure helpers for returned-result filtering, quote inspection markup, extraction presentation, document-page boundaries, per-operation latency, and common retrieval comparisons.

- [ ] Confirm `origin/main`, `origin/phase12/api-serving-boundary`, current PR head, and tree SHAs; do not rewrite history yet.
- [ ] Add failing tests for document filtering preserving `Evidence.rank`, exact quote highlighting in a surrounding `DocumentUnit`, pagination boundaries, structured extraction rows, endpoint-family latency summaries, and common retrieval comparison rows.
- [ ] Run the focused tests and verify they fail because the helpers do not yet exist.

### Task 2: Implement evidence and extraction presentation helpers

**Files:**
- Modify: `src/kawaneen/ui/formatting.py`, `src/kawaneen/ui/evaluation.py`, `src/kawaneen/ui/exports.py`
- Test: `tests/test_ui_formatting.py`, `tests/test_ui_evaluation.py`, `tests/test_ui_exports.py`

**Interfaces:**
- `filter_returned_evidence(results, selected_documents) -> tuple[Evidence, ...]` filters only returned evidence and retains tuple order/ranks.
- `inspect_verified_quote(unit, quote) -> str` returns escaped literal surrounding-unit HTML with a safe mark around an exact match; no fuzzy matching.
- `extract_presentation_rows(segment_id, response) -> tuple[dict[str, object], ...]` exposes counts, structured spans, source identity, and segment identity.
- `common_retrieval_comparison(snapshot) -> tuple[dict[str, object], ...]` returns only metrics present across compared tracked models/splits.
- `aggregate_latency_by_operation(values_by_operation) -> dict[str, LatencySummary]` keeps Search, Answer, and Extract separate and excludes demo values by caller contract.

- [ ] Implement the minimum pure helpers needed for the failing tests.
- [ ] Run focused helper tests until green.
- [ ] Refactor only after green; keep all dynamic text escaped and JavaScript-free.

### Task 3: Upgrade Search, Ask, and Extract pages

**Files:**
- Modify: `src/kawaneen/ui/pages/search.py`, `src/kawaneen/ui/pages/ask.py`, `src/kawaneen/ui/pages/extract.py`, `src/kawaneen/ui/components.py`, `src/kawaneen/ui/state.py`
- Test: `tests/test_ui_app.py`, `tests/test_ui_aaa_render_coverage.py`, `tests/test_ui_state.py`

**Interfaces:**
- Search document multiselect options are derived from the current `response.results` only; the displayed tuple remains API-ranked.
- Ask uses the exact `locate_quote` result and renders the canonical unit around the quote plus metadata and conditional source link.
- Extract stores `DocumentPage` offset/limit/total state, calls `list_documents(offset, limit)`, and renders `Documents X–Y of TOTAL` with Prev/Next controls.

- [ ] Add AppTest/hermetic tests for filtering, citation inspection, pagination controls, and structured presentation.
- [ ] Run those tests to verify the new assertions fail.
- [ ] Implement the page changes with safe HTML, `st.html()` where available, and no JavaScript.
- [ ] Run AppTest and focused page tests until green.

### Task 4: Upgrade the evaluation dashboard and README

**Files:**
- Modify: `src/kawaneen/ui/pages/evaluation.py`, `src/kawaneen/ui/evaluation.py`, `README.md`
- Test: `tests/test_ui_evaluation.py`, `tests/test_ui_app.py`, `tests/test_ui_aaa_render_coverage.py`

**Interfaces:**
- Evaluation renders capability provider/model/short revision/readiness, frozen architecture text, common tracked retrieval comparisons/deltas, generation cards, extraction cards/error chart, separate live Search/Answer/Extract latency, and collapsed source hashes.

- [ ] Add failing dashboard data/latency tests where needed.
- [ ] Implement the recruiter-grade layout using only tracked snapshot data and current session state.
- [ ] Add the Search screenshot prominently and the remaining screenshot references to README after screenshots exist; otherwise document the visual gate accurately.
- [ ] Run AppTest and evaluation tests until green.

### Task 5: Run code/test gates and perform browser tooling discovery

**Files:**
- Modify: `docs/reports/phase-13-ui-report.md`
- Create: `docs/assets/ui/search.png`, `docs/assets/ui/ask.png`, `docs/assets/ui/extract.png`, `docs/assets/ui/evaluation.png` only after real rendered inspection succeeds

- [ ] Run `ruff format`, `ruff check`, `pyright`, Streamlit AppTest, public pytest with branch coverage >=85%, and targeted private smoke.
- [ ] Start `make ui-demo` and check `agent-browser` availability first.
- [ ] If unavailable, explicitly check existing Chromium/Chrome/Playwright-capable executables without adding runtime dependencies.
- [ ] At 1440×900, 1280×800, and one narrow viewport inspect Arabic/English Search, grounded/abstention Ask, Extract, and Evaluation for hierarchy, RTL, clipping, overflow, density, labels, and empty/error states.
- [ ] Capture only synthetic screenshots at the four required paths, or record exact tooling unavailability and leave the screenshot gate open.

### Task 6: Rebase, retarget PR, and verify fresh CI

**Files:**
- Modify: Git history and PR metadata only; no Phase 7–12 source/artifact edits.

- [ ] Commit product/test/docs changes and record old head.
- [ ] Run `git fetch origin` and `git rebase --onto origin/main ac6d8f8e079cfe88b3a4d5c5f06977eb68f17fc2 phase13/streamlit-product-interface`.
- [ ] Treat substantive conflicts as suspicious; resolve only Phase 13 files.
- [ ] Verify `git merge-base HEAD origin/main == origin/main`, Phase 13-only diff, clean status, and unchanged frozen trees.
- [ ] Push only with `git push --force-with-lease origin phase13/streamlit-product-interface`.
- [ ] Retarget PR #9 to `main`, confirm CLEAN/MERGEABLE, and require/read the fresh Python 3.11 and 3.12 pull-request CI jobs on the rewritten head.
- [ ] Do not merge.

