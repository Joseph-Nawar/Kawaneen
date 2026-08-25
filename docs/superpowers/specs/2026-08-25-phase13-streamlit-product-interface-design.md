# Phase 13 Streamlit Product Interface Design

## Goal

Add a recruiter-facing, evidence-first Streamlit workspace for Saudi legal research and document intelligence. The interface is a presentation and orchestration boundary over the Phase 12 HTTP API; it does not call retrieval, generation, extraction, or corpus runtime services directly.

## Scope and guardrails

- Four screens: Search, Ask, Extract, and Evaluation.
- Streamlit `>=1.61,<2`, wide layout, top `st.navigation`, and restrained theme configuration.
- API base URL defaults to `http://127.0.0.1:8000`.
- Runtime mode is explicit: `KAWANEEN_UI_MODE=auto|live|demo`.
- `auto` probes `/v1/health`; it never presents demo results as live and requires an explicit user action to enter demo mode when live readiness is unavailable.
- Demo fixtures and screenshots are small, tracked, synthetic Arabic/English examples only.
- No OCR, persistence, document parsing beyond `.txt`, `.md`, and text-based PDFs through `pypdf`, no private paths, secrets, raw stack traces, provider payloads, or fake confidence scores.
- Search scope is jurisdiction-only (`SA`); returned-evidence refinement is visually distinct and does not invent unavailable metadata filters.
- Hybrid extraction visibly carries `PHASE11_HYBRID_EXPERIMENTAL_LIMITED` and its explanation.
- Evaluation data is generated only from tracked, sanitized Phase 8/10/11 sources after provenance/hash validation. HOLDOUT content and private artifacts are never read.

## Architecture

`src/kawaneen/ui/app.py` is the only Streamlit entrypoint. It configures the page and registers `st.Page` objects for the four screens. Page modules depend on small, pure helpers and on a `UiServices` session object; they do not import runtime-serving modules.

The typed HTTP client wraps `httpx.Client` and validates successful JSON with existing `kawaneen.api.contracts` models. It maps timeout, connection, HTTP, and validation failures into safe `UiApiError` values. A demo client implements the same protocol with synthetic Pydantic responses, so AppTest never needs model or corpus assets.

Pure UI helpers cover RTL detection, safe literal highlighting, citation quote location, upload validation, paragraph/unit segmentation, JSON/CSV export, and latency aggregation. Evaluation snapshot code reads an allowlisted set of tracked metadata/metrics files, rejects paths outside the repository or files with disallowed/private markers, and records SHA-256 hashes next to the sanitized values.

## Screen behavior

### Search

The query form accepts Arabic and English text, exposes Saudi Arabia as the only jurisdiction, and allows the API-supported result limit. Results show rank, title, article/page metadata, provenance, and a literal-highlighted evidence excerpt. Raw reranker logits are visible only in a collapsed technical section and are never described as confidence. A returned-evidence refinement input filters the displayed result set while preserving original rank and labels the scope explicitly.

### Ask

The page is a 60/40 answer/evidence layout. Answerable responses show the grounded answer, citation cards, exact verified quotes, source metadata, and optional source links only when present. Citation inspection can fetch document details and locate the quote in canonical units. Abstentions render an amber intentional-safety state with the human-readable reason and available evidence, not an error or chat transcript.

### Extract

The source selector supports paste text, a bounded upload flow, and paginated corpus document selection. Text is segmented on paragraph/unit boundaries at no more than 18,000 characters, with at most five segments and no silent truncation. Each segment retains an identity in results and exports. Deterministic and hybrid modes map to `/v1/extract`; hybrid output includes the required experimental label. Results group obligations, deadlines, regulated entities, exceptions, and other structured fields without confidence language. JSON and flattened CSV downloads are available.

### Evaluation

The page shows live `/v1/models` readiness, frozen retrieval configuration/comparisons and selected deltas, tracked Phase 10 DEV metrics with their DEV/AI-reviewed qualification, protected Phase 11 HOLDOUT summary values with the source provenance statement, an error-taxonomy chart, and current-session latency labelled as non-benchmark. It never uses demo latency as live latency.

## Visual system

The brand is `KAWANEEN | قوانين` with subtitle `Arabic Legal Intelligence · Saudi Arabia`. The design uses a warm off-white canvas, white surfaces, deep navy text, muted teal primary, restrained gold accents, subtle borders, 8–10px radii, and nearly no shadows. Static CSS is escaped and injected through `st.html`; user/legal text is escaped before custom markup. Arabic blocks use RTL direction and mixed Arabic/English text remains readable without clipping or horizontal overflow.

## Verification

- Unit tests use `httpx.MockTransport` for live-client behavior.
- Streamlit `AppTest` runs all four pages and key demo interactions.
- A private-artifact smoke is optional, narrow, and never invokes evaluation/HOLDOUT/tuning endpoints.
- `make check`, coverage of at least 85%, and the existing Python 3.11/3.12 CI matrix must pass.
- Browser QA covers 1440x900, 1280x800, and one narrow viewport for Arabic search, English search, answer, abstention, extraction, and evaluation; screenshots use demo fixtures only.
- Final audit verifies no private legal text, screenshots containing private data, secrets, machine paths, caches, uploads, HOLDOUT reads, or frozen Phase 7–12 changes.
