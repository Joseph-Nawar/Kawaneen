# Phase 12 API Serving Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-safe FastAPI `/v1` serving boundary for search, grounded answers, extraction, canonical documents, health, and model capability metadata.

**Architecture:** The HTTP package owns contracts, middleware, error mapping, dependency injection, and request lifecycle only. Serving adapters expose narrow synchronous lower-level interfaces for retrieval, generation/grounding, extraction, and canonical corpus access; the API executes them in worker threads with bounded timeouts. Default runtime construction is readiness-aware and lazy about optional heavyweight assets, while hermetic tests inject fake adapters.

**Tech Stack:** Python 3.11/3.12, FastAPI, Uvicorn, Pydantic v2, AnyIO, structlog, httpx, pytest.

**Spec:** User-provided Phase 12 API serving boundary requirements in the task request.

## Global Constraints

- Support Python `>=3.11,<3.13` and use `uv` for synchronization and commands.
- Phase 0 remains foundation only; do not add parsing, retrieval, RAG, APIs outside this Phase 12 boundary, Docker, secrets, datasets, or network-dependent tests.
- All public API models use `extra="forbid"`; v1 supports jurisdiction `SA` only.
- Search/answer queries are capped at 2,000 characters; extraction text at 20,000; request IDs at 128 safe characters; POST bodies at 128 KiB.
- The serving layer must not call DEV/evaluation orchestrators, qrels, protected queries, HOLDOUT, or private artifact readers for ordinary requests.
- Frozen retrieval configuration is sparse top-50 + dense top-50 + RRF + 20 candidates + reranker + depth 8, with raw reranker logits.
- No complete query or extraction text is logged, and no stack traces, paths, prompts, provider bodies, or private artifact details are exposed.
- Run `make check` before handoff; do not add Phase 12 files to coverage omissions.

---

### Task 1: Serving contracts, errors, and request context

**Files:**
- Create: `src/kawaneen/api/__init__.py`
- Create: `src/kawaneen/api/contracts.py`
- Create: `src/kawaneen/api/errors.py`
- Create: `src/kawaneen/api/context.py`
- Test: `tests/test_api_contracts.py`

**Interfaces:**
- Produces strict request/response models for all seven endpoints, bounded fields, `SA` jurisdiction, structured `ApiError`, and request-ID sanitization/context helpers.
- Later routers consume `SearchRequest`, `AnswerRequest`, `ExtractRequest`, document/page models, readiness/model models, and `ApiError`.

- [ ] **Step 1: Write failing contract tests** for extra fields, query/text/limit bounds, jurisdiction rejection, valid request IDs, and structured error serialization.
- [ ] **Step 2: Run `uv run pytest tests/test_api_contracts.py -q`** and confirm failures are due to missing API contracts.
- [ ] **Step 3: Implement the strict Pydantic v2 models and safe request-ID regex** with no filesystem or provider imports.
- [ ] **Step 4: Re-run the focused tests** and confirm they pass.
- [ ] **Step 5: Commit** with `git add src/kawaneen/api tests/test_api_contracts.py && git commit -m "feat: add phase 12 api contracts"`.

### Task 2: Serving adapters and injectable runtime container

**Files:**
- Create: `src/kawaneen/retrieval/serving.py`
- Create: `src/kawaneen/generation/serving.py`
- Create: `src/kawaneen/extraction/serving.py`
- Create: `src/kawaneen/corpus/serving.py`
- Create: `src/kawaneen/api/runtime.py`
- Test: `tests/test_api_serving_adapters.py`

**Interfaces:**
- `ServingRetriever.search(query: str, limit: int) -> RetrievalResponse` must call only sparse/dense lower-level indexes, `FusionConfig()` and frozen reranker contracts.
- `ServingAnswerer.answer(query: str) -> AnswerPipelineResult` composes retrieval, `ContextAssembler`, `evaluate_stage_d_policy`, generation, `verify_draft`, and fail-closed finalization without DEV runners.
- `ServingExtractor.extract(text: str, mode: ExtractionMode) -> ExtractionResponse` uses `run_deterministic` and, for hybrid, an injected `ExtractionProvider` plus `assemble_hybrid_result`.
- `ServingCorpus.list_documents(offset: int, limit: int)` and `.get_document(document_id)` return path-free canonical metadata/units.
- `ServiceContainer` owns adapters, readiness/model snapshots, async-safe lifecycle initialization and idempotent shutdown; `build_default_container(settings)` has no inference at import and records expected missing assets as not-ready.

- [ ] **Step 1: Write failing fake-provider tests** for frozen retrieval depth/raw logits, deterministic zero provider calls, hybrid validation/no fallback, answer policy-before-generation, citation fail-closed behavior, and document path sanitization.
- [ ] **Step 2: Run the focused adapter tests** and confirm missing serving modules/interfaces fail.
- [ ] **Step 3: Implement narrow protocols and adapters**, reusing only lower-level Phase 7–11 primitives and keeping all provider calls injectable.
- [ ] **Step 4: Add readiness metadata and cleanup hooks**, ensuring repeated `initialize()` loads once and `close()` is idempotent.
- [ ] **Step 5: Re-run focused tests** and commit the serving boundary.

### Task 3: Application factory, lifespan, middleware, dependency injection, and error mapping

**Files:**
- Create: `src/kawaneen/api/app.py`
- Create: `src/kawaneen/api/dependencies.py`
- Create: `src/kawaneen/api/middleware.py`
- Create: `src/kawaneen/api/routers.py`
- Test: `tests/test_api_app.py`

**Interfaces:**
- `create_app(container_factory: Callable[[], ServiceContainer] | None = None, settings: Settings | None = None) -> FastAPI` exposes `/v1` routes and OpenAPI models with unique operation IDs.
- Lifespan initializes exactly once and closes the injected container exactly once.
- Middleware enforces request-ID generation/echo/context isolation, POST body limits, safe logging, and maps domain/timeouts/unexpected exceptions to `ApiError` status codes.

- [ ] **Step 1: Write failing ASGI tests** for all paths, request IDs, body/query limits, 404/413/422/503/504/500 envelopes, timeout conversion, context cleanup, lifespan counts, readiness/degraded state, and `/models` no-load behavior.
- [ ] **Step 2: Run `uv run pytest tests/test_api_app.py -q`** and verify the expected missing-app failures.
- [ ] **Step 3: Implement the FastAPI factory, lifespan, dependency override point, and middleware** using `anyio.to_thread.run_sync` plus `asyncio.timeout`/AnyIO timeout handling around synchronous services.
- [ ] **Step 4: Implement router handlers** that call services only through dependencies and return contract models with request IDs.
- [ ] **Step 5: Re-run focused tests** and commit.

### Task 4: CLI/runtime dependencies, OpenAPI examples, and documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/kawaneen/cli.py`
- Modify: `Makefile`
- Modify: `README.md`
- Create: `docs/api.md`
- Create: `docs/reports/phase-12-api-report.md`
- Test: `tests/test_api_cli_docs.py`

**Interfaces:**
- `uv run kawaneen api serve [--host HOST] [--port PORT]` defaults to `127.0.0.1:8000` and delegates to Uvicorn.
- `make api-serve` invokes the same default command.
- Documentation describes request/response examples, limits, errors, readiness, and no-private-data guarantees.

- [ ] **Step 1: Write failing CLI/OpenAPI/documentation tests** for the command shape, default bind, required paths, operation-ID uniqueness, and examples.
- [ ] **Step 2: Run the focused tests** and verify they fail before dependency/CLI changes.
- [ ] **Step 3: Add FastAPI/Uvicorn/httpx runtime/dev dependencies and regenerate the lock with `uv lock`**, then add CLI and Make targets.
- [ ] **Step 4: Add the API docs and phase report skeleton with explicit verification checklist.**
- [ ] **Step 5: Re-run focused tests and commit.

### Task 5: Full public verification and review gates

**Files:**
- Modify: `tests/` only for missing public hermetic coverage discovered by verification.
- Modify: `.github/workflows/` only if the existing matrix cannot run Python 3.11/3.12 public checks.
- Modify: `docs/reports/phase-12-api-report.md` with measured results.

**Interfaces:**
- Public tests cover every requirement in the Phase 12 request without `private_artifact` markers; private smoke tests, if present, are narrowly marked and never run in public verification.

- [x] **Step 1: Run `uv run ruff format --check .`, `uv run ruff check .`, `uv run pyright`, and `uv run pytest -m 'not private_artifact' --cov=kawaneen --cov-branch --cov-report=term-missing --cov-fail-under=85`.**
- [x] **Step 2: Run `make check` and the clean-checkout/public-CI equivalent without Qwen/Ollama, HOLDOUT, or evaluation runners.**
- [ ] **Step 3: Audit tracked files for absolute workstation paths, secrets, private artifacts, model outputs, caches, and raw inputs; compare frozen Phase 10/11 tracked hashes/results before and after.**
- [ ] **Step 4: Request a code review, fix Critical/Important findings, then commit the final implementation.**
- [ ] **Step 5: Push `phase12/api-serving-boundary` and open/update the PR without merging; record exact tests, coverage, lint/type results, commit SHA, PR status, CI jobs, and safety confirmations in the report/final response.**

## Self-review checklist

- The plan covers all seven endpoints, strict contracts, limits, request IDs, structured errors, timeouts, async safety, serving adapters, lifecycle/readiness/models, CLI/docs, tests, coverage, CI, and no-private-artifact audit.
- No implementation task calls a Phase 8 DEV orchestrator, `run_dev_generation`, Phase 11 evaluation runner, qrels, protected queries, or HOLDOUT.
- Adapter signatures are independent of HTTP and can be backed by deterministic fakes for every public test.
- All model and error payloads are path-free and avoid prompts/provider bodies/query logging.
