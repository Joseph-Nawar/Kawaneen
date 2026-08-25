# Phase 12 API Serving Boundary Report

Status: implementation branch `phase12/api-serving-boundary`; PR [#8](https://github.com/Joseph-Nawar/Kawaneen/pull/8) is open and intentionally not merged by the implementation agent.

## Architecture

The new `src/kawaneen/api/` package contains strict Pydantic contracts, structured errors, request-ID/body-limit middleware, FastAPI application factory/lifespan, dependency accessors, routers, and the injectable `ServiceContainer`. Serving-safe adapters live in `retrieval/serving.py`, `generation/serving.py`, `extraction/serving.py`, and `corpus/serving.py`.

The answer adapter binds the lower-level Phase 9 `ContextAssembler` and citation verifier and Phase 10 answerability policy through `ServingAnswerer.from_phase9_10`; it does not invoke DEV or evaluation orchestrators. Retrieval uses the frozen hybrid contracts directly. Extraction uses Phase 11 deterministic extraction and `assemble_hybrid_result` with request-scoped provenance.

## Contract and lifecycle summary

The API exposes `/v1/search`, `/v1/answer`, `/v1/extract`, `/v1/documents`, `/v1/documents/{document_id}`, `/v1/health`, and `/v1/models`. All public models forbid extra fields, v1 accepts only `SA`, and the documented query, text, body, page, and request-ID limits are enforced. Lifespan initialization and cleanup are idempotent; missing expected local assets are represented as degraded readiness, while programming/configuration failures remain visible.

## Verification record

- Public tests: 774 passed, 1 skipped, 38 deselected (`uv run pytest -m "not private_artifact" ...`)
- Branch coverage: 85.12% (required threshold 85%)
- Ruff format/lint: clean (`uv run ruff format --check .`; `uv run ruff check .`)
- Pyright: clean (`0 errors, 0 warnings, 0 informations`)
- Implementation commit SHA: `b1c5f5b`
- Python 3.11 / 3.12 GitHub Actions jobs: pending GitHub run creation
- PR: #8, open, not merged

Safety confirmations for the final run:

- No Qwen/Ollama calls: required
- No HOLDOUT/protected-query access: required
- Phase 10 and Phase 11 frozen results/metrics unchanged: required
- No private artifacts, model outputs/caches, secrets, or machine-specific paths committed: required
