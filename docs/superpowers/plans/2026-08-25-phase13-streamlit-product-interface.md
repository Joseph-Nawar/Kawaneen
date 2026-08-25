# Phase 13 Streamlit Product Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished, evidence-first Streamlit product interface over the Phase 12 `/v1` HTTP API with safe demo mode, tracked evaluation evidence, tests, and browser-verified screenshots.

**Architecture:** `kawaneen.ui.app` owns Streamlit page registration only. Focused UI modules provide a typed HTTP client, session/mode state, pure formatting/upload/export helpers, sanitized evaluation snapshots, shared components, CSS, and four page modules. Demo services implement the same client protocol with synthetic Pydantic responses so AppTest remains hermetic.

**Tech Stack:** Python 3.11–3.12, Streamlit `>=1.61,<2`, httpx, existing Pydantic Phase 12 contracts, pypdf for text PDFs, pytest/AppTest, Ruff, Pyright.

**Spec:** `docs/superpowers/specs/2026-08-25-phase13-streamlit-product-interface-design.md`

## Global Constraints

- Streamlit consumes the Phase 12 `/v1` HTTP API only.
- `auto` mode never silently presents demo data as live.
- All demo fixtures and screenshots are synthetic.
- Search is evidence-first and exposes no unsupported metadata filters or fake confidence scores.
- Hybrid extraction always displays `PHASE11_HYBRID_EXPERIMENTAL_LIMITED`.
- Evaluation values come only from sanitized, provenance-checked tracked sources.
- Do not access HOLDOUT, rerun evaluations, tune models, or alter frozen Phase 7–12 results.
- User/legal text must be escaped before custom HTML; no JavaScript or `st.chat_*`.
- API imports and module loading remain free of filesystem/network side effects.
- `make check` is required before handoff.

---

### Task 1: UI dependency, settings, mode state, and public entrypoint

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `Makefile`
- Create: `.streamlit/config.toml`
- Create: `src/kawaneen/ui/__init__.py`
- Create: `src/kawaneen/ui/config.py`
- Create: `src/kawaneen/ui/state.py`
- Create: `src/kawaneen/ui/app.py`
- Test: `tests/test_ui_config.py`

**Interfaces:**
- `UiSettings.from_env(environ: Mapping[str, str] | None = None) -> UiSettings` reads `KAWANEEN_UI_MODE`, `KAWANEEN_API_URL`, and bounded timeout settings.
- `UiMode` is the literal enum `auto`, `live`, `demo`.
- `resolve_mode(settings: UiSettings, health: HealthProbe | None) -> ModeResolution` returns the active mode, status label (`Live API`, `Demo data`, or `Degraded`), and an explicit demo-activation requirement.
- `make ui-serve` runs `uv run streamlit run -m kawaneen.ui.app`; `make ui-demo` sets `KAWANEEN_UI_MODE=demo` and runs the same command.

- [ ] **Step 1: Write failing configuration tests.**

```python
def test_auto_mode_requires_explicit_demo_activation_when_api_is_unavailable():
    settings = UiSettings.from_env({"KAWANEEN_UI_MODE": "auto"})
    resolution = resolve_mode(settings, HealthProbe.unavailable("connection refused"))
    assert resolution.active_mode is None
    assert resolution.requires_demo_activation is True
    assert resolution.status_label == "Degraded"
```

- [ ] **Step 2: Run `uv run pytest tests/test_ui_config.py -q` and verify it fails because the UI modules do not exist.**
- [ ] **Step 3: Add the optional dependency, lock entry, settings/state modules, theme config, and Make targets.** Keep import-time behavior pure; do not create a Streamlit client or call the network during import.
- [ ] **Step 4: Implement `app.py` with wide layout and top navigation registering four page callables.** Page imports should be lazy enough that importing helpers remains safe in tests.
- [ ] **Step 5: Run the focused tests and `uv run python -c 'import kawaneen.ui.app'`.**
- [ ] **Step 6: Commit `feat: scaffold phase 13 ui boundary`.**

### Task 2: Typed API client, safe errors, and synthetic demo fixtures

**Files:**
- Create: `src/kawaneen/ui/client.py`
- Create: `src/kawaneen/ui/demo.py`
- Test: `tests/test_ui_client.py`
- Test: `tests/test_ui_demo.py`

**Interfaces:**
- `UiClient.search(request: SearchRequest) -> SearchResponse`.
- `UiClient.answer(request: AnswerRequest) -> AnswerResponse`.
- `UiClient.extract(request: ExtractRequest) -> ExtractionResponse`.
- `UiClient.list_documents(offset: int, limit: int) -> DocumentPage`.
- `UiClient.get_document(document_id: str) -> DocumentDetail`.
- `UiClient.health() -> HealthResponse` and `.models() -> ModelsResponse`.
- `UiApiError` exposes only `code`, `message`, `status_code`, and optional `request_id`.
- `DemoClient` implements the same methods with synthetic response models and four scenarios.

- [ ] **Step 1: Write failing MockTransport tests** for strict response validation, safe error envelopes, timeout mapping, request IDs, and no raw response leakage.
- [ ] **Step 2: Run `uv run pytest tests/test_ui_client.py tests/test_ui_demo.py -q` and verify expected failures.**
- [ ] **Step 3: Implement the client with `httpx.Client`, Pydantic `model_validate`, bounded timeouts, and safe exception mapping.** Never import API runtime composition.
- [ ] **Step 4: Add synthetic search, grounded-answer, abstention, extraction, document, health, and models fixtures.** Ensure demo latency is marked synthetic and is never appended to live session latency.
- [ ] **Step 5: Run focused tests and commit `feat: add typed phase 12 ui client`.**

### Task 3: Pure safety, formatting, segmentation, citation, and export helpers

**Files:**
- Create: `src/kawaneen/ui/formatting.py`
- Create: `src/kawaneen/ui/uploads.py`
- Create: `src/kawaneen/ui/exports.py`
- Test: `tests/test_ui_formatting.py`
- Test: `tests/test_ui_uploads.py`
- Test: `tests/test_ui_exports.py`

**Interfaces:**
- `contains_arabic(text: str) -> bool` and `text_direction(text: str) -> Literal["rtl", "ltr"]`.
- `highlight_literal(text: str, query: str) -> str` returns escaped HTML with literal, case-insensitive matches marked safely.
- `locate_quote(units: Sequence[DocumentUnit], quote: str) -> QuoteLocation | None` returns canonical unit and codepoint offsets.
- `validate_upload(name: str, size_bytes: int, allowed_bytes: int) -> UploadDecision`.
- `extract_text(name: str, payload: bytes) -> str` supports text/markdown and text PDFs with `pypdf`, rejecting scanned/no-readable-text PDFs with safe messages.
- `segment_text(text: str, max_chars: int = 18_000, max_segments: int = 5) -> tuple[TextSegment, ...]` never silently truncates and preserves segment IDs.
- `extraction_json(response: ExtractionResponse | Sequence[ExtractionResponse]) -> bytes` and `extraction_csv(...) -> bytes` produce deterministic downloads.

- [ ] **Step 1: Write failing tests** for XSS/HTML escaping, Arabic/mixed text, literal highlighting, exact quote offsets, extension/size/PDF handling, paragraph boundaries, 20k API limits, max five segments, JSON/CSV segment identity, and empty exports.
- [ ] **Step 2: Run each focused test file and verify failures before implementation.**
- [ ] **Step 3: Implement pure helpers with standard library plus `pypdf` only in the upload path.** Do not write uploaded bytes to disk.
- [ ] **Step 4: Run focused tests, then refactor while green.**
- [ ] **Step 5: Commit `feat: add safe ui formatting and document helpers`.**

### Task 4: Sanitized evaluation snapshot and session latency

**Files:**
- Create: `src/kawaneen/ui/evaluation.py`
- Create: `data/manifests/ui/phase13_evaluation_snapshot.json`
- Test: `tests/test_ui_evaluation.py`

**Interfaces:**
- `build_evaluation_snapshot(root: Path) -> EvaluationSnapshot` reads only an explicit allowlist of tracked Phase 8/10/11 files, validates repository-relative paths and SHA-256 values, and returns sanitized metrics plus provenance records.
- `write_evaluation_snapshot(root: Path, destination: Path) -> None` writes stable JSON under `data/manifests/ui/`.
- `aggregate_latency(values: Sequence[float], limit: int = 50) -> LatencySummary` keeps the most recent bounded values and computes min/median/p95/max without calling benchmark artifacts.

- [ ] **Step 1: Write failing tests** for source allowlisting, SHA-256 provenance, stable metric extraction, rejection of private/HOLDOUT paths, and last-50 latency aggregation.
- [ ] **Step 2: Run `uv run pytest tests/test_ui_evaluation.py -q` and verify expected failures.**
- [ ] **Step 3: Implement the snapshot builder from tracked sanitized sources only.** Preserve the required Phase 10 DEV and Phase 11 HOLDOUT summary labels and poor metrics; do not read protected result contents or private paths.
- [ ] **Step 4: Generate the tracked snapshot and test its hashes.**
- [ ] **Step 5: Commit `feat: add provenance-checked ui evaluation snapshot`.**

### Task 5: Shared visual components and page implementations

**Files:**
- Create: `src/kawaneen/ui/styles.py`
- Create: `src/kawaneen/ui/components.py`
- Create: `src/kawaneen/ui/pages/__init__.py`
- Create: `src/kawaneen/ui/pages/search.py`
- Create: `src/kawaneen/ui/pages/ask.py`
- Create: `src/kawaneen/ui/pages/extract.py`
- Create: `src/kawaneen/ui/pages/evaluation.py`
- Modify: `src/kawaneen/ui/app.py`
- Test: `tests/test_ui_app.py`

**Interfaces:**
- `render_product_header(state: UiSessionState) -> None`, `render_status_banner(...) -> None`, `render_evidence_card(...) -> None`, `render_warning_list(...) -> None`, and `render_citation_card(...) -> None` are reusable Streamlit components.
- Each page exposes `render(client: UiClient, state: UiSessionState) -> None`.

- [ ] **Step 1: Write failing AppTest tests** for page registration, status labels, Search Arabic/English interactions, evidence inspector/refinement, grounded answer citations, abstention state, Extract source modes/segmentation/hybrid label/downloads, and Evaluation sections.
- [ ] **Step 2: Run `uv run pytest tests/test_ui_app.py -q` and verify failures.**
- [ ] **Step 3: Implement the visual system and shared components.** Use escaped static CSS only, no JavaScript, no chat components, and preserve RTL direction and wrapping.
- [ ] **Step 4: Implement Search, Ask, Extract, and Evaluation pages against the client protocol.** Keep the API call in the page action, not import time; display scope and evidence semantics explicitly.
- [ ] **Step 5: Run AppTest and focused unit tests; fix any state-key collisions or rerun issues.**
- [ ] **Step 6: Commit `feat: implement recruiter-facing ui pages`.**

### Task 6: README, screenshots, browser QA, and private smoke

**Files:**
- Modify: `README.md`
- Create: `docs/assets/ui/search.png`
- Create: `docs/assets/ui/ask.png`
- Create: `docs/assets/ui/extract.png`
- Create: `docs/assets/ui/evaluation.png`
- Create: `tests/test_ui_private_smoke.py`
- Modify: `docs/reports/phase-13-ui-report.md`

**Interfaces:**
- README documents `make api-serve` + `make ui-serve` for live use and `make ui-demo` for synthetic demo mode.
- The private smoke is marked `private_artifact`, skips unless local Phase 12 assets exist, and calls only health/models plus one normal UI request; it never uses HOLDOUT/evaluation/tuning endpoints.

- [ ] **Step 1: Write the smoke test and documentation assertions.**
- [ ] **Step 2: Run the smoke test without private assets and verify it skips cleanly.**
- [ ] **Step 3: Start `make ui-demo` and use available browser tooling to inspect 1440x900, 1280x800, and a narrow viewport.** Check Arabic search, English search, answer, abstention, extraction, evaluation, wrapping, clipping, empty states, and status labels.
- [ ] **Step 4: Fix visible defects, then capture the four synthetic screenshots at stable viewport sizes.**
- [ ] **Step 5: Run the complete verification suite:**

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -m "not private_artifact" --cov=kawaneen --cov-branch --cov-report=term-missing --cov-fail-under=85
make check
```

- [ ] **Step 6: Audit the diff and tracked files** for private data, screenshots with private content, secrets, machine paths, caches, uploads, HOLDOUT reads, and frozen Phase 7–12 modifications.
- [ ] **Step 7: Push `phase13/streamlit-product-interface`, open the Phase 13 PR without merging, and record the exact SHA, PR state, CI jobs, test/coverage/AppTest/browser results in `docs/reports/phase-13-ui-report.md`.**

## Plan self-review

- Search, Ask, Extract, Evaluation, demo/live mode, API-only boundary, safe markup, uploads, segmentation, exports, snapshot hashes, latency, tests, visual QA, screenshots, README, Make targets, private smoke, CI, and final audit all have explicit tasks.
- No task reads HOLDOUT content, runs evaluation, tunes models, or changes frozen Phase 7–12 results.
- Public tests can run without model/corpus assets; only the marked smoke depends on private local state.
- The plan names concrete modules, interfaces, commands, and test behaviors without placeholders.
