# Phase 14 Testing Hardening Implementation Plan

Status: completed on `phase14/testing-hardening`; verification details are in
`docs/reports/phase-14-testing-report.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing 877-test Kawaneen suite into explicit unit, deterministic integration, regression, optional model-artifact, and Docker Compose E2E layers without changing Phase 7–13 runtime behavior.

**Architecture:** Reuse the shipped parsing, normalization, chunking, NumPy exact index, BM25/RRF serving, grounding, answerability, citation, FastAPI, and Streamlit primitives. Keep synthetic fixtures and regression metadata public and text-only; use a test-only Compose image with lightweight deterministic adapters and dependency injection.

**Tech Stack:** Python 3.11/3.12, uv, pytest/pytest-cov, Ruff, Pyright, FastAPI TestClient, Streamlit AppTest where available, Docker Compose, official Python 3.12 slim arm64/amd64 image.

**Spec:** User-provided Phase 14 brief, anchored at authoritative main `294c8c171abc0e93d826ffbb6d2cf019c63e6d44`.

## Global Constraints

- Base branch is `main` at `294c8c171abc0e93d826ffbb6d2cf019c63e6d44`; work on `phase14/testing-hardening`; do not merge.
- Public fixtures and regression cases are synthetic only; do not access HOLDOUT, private corpora, secrets, model downloads, or machine-local paths.
- Public retrieval uses the existing `NumpyExactIndex`; no Qdrant, Redis, Postgres, CUDA, Ollama, or cloud services.
- Public E2E uses deterministic adapters and native multi-architecture Docker; no `platform: linux/amd64`.
- Production imports remain free of filesystem and network side effects; Docker is a test harness only.
- Use Python `>=3.11,<3.13), uv, TDD for behavior changes, and run `make check` before handoff.

## Baseline test-gap matrix

| Phase 14 requirement | Existing coverage on authoritative main | Gap to add |
| --- | --- | --- |
| Arabic normalization and idempotence | `tests/test_normalization_*.py`, existing synthetic cases | Marker/layer placement and a compact invariant audit |
| Article extraction and malformed references | `tests/test_corpus_statutory.py` | Explicit reference false-positive matrix |
| Metadata/schema contracts | Pydantic contract tests across corpus, retrieval, generation, grounding, API | Layered schema fail-closed audit |
| Chunk boundaries/provenance | `tests/test_chunking_*.py` | Integration assertions against one parsed synthetic PDF |
| Weighted RRF and serving depths | `tests/test_retrieval_hybrid_*.py`, `tests/test_api_serving_adapters.py` | Public integration path and config lock |
| Citation/abstention | `tests/test_grounding_*.py`, `tests/test_generation_*.py`, `tests/test_api_answer_serving.py` | End-to-end mutation matrix and generator boundary |
| PDF to canonical/chunks | No public fixture-backed pipeline | Add tiny machine-readable synthetic PDF and integration fixture |
| Chunks to real NumPy index | Unit vector-index tests exist | Add integration artifact correspondence/ranking test |
| Query to answer/display | API/UI tests use independent synthetic fixtures | Add one wired deterministic answer and UI display path |
| Regression behavior lock | No Phase 14 lock/case suite | Add 20 public cases, lock, hash/config verification |
| Markers/commands | Only `private_artifact` marker and broad commands | Standard markers and dedicated Make targets |
| Real model regression | No narrowly marked tier | Cache-only BGE tests with explicit skip reason |
| Docker public E2E | None | Test-only Compose image, runner, healthcheck, cleanup |
| CI quality/e2e separation | One matrix quality job | Preserve 3.11/3.12 quality and add one Ubuntu Compose job |
| Testing documentation | `docs/testing-policy.md` exists | Add Phase 14 guide/report and minimal README links |

---

### Task 1: Establish test-layer markers and commands

**Files:**
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_phase14_test_layers.py`

**Interfaces:**
- Produces pytest markers `integration`, `regression`, `model_artifact`, `e2e`, and existing `private_artifact` semantics.
- Produces Make targets `test-unit`, `test-integration`, `test-regression`, `test-model-regression`, `test-e2e`, `test-e2e-private`, `test-public`, and `check`.

- [ ] **Step 1: Write failing marker/command tests** asserting marker names are registered and the Makefile contains each target.
- [ ] **Step 2: Run `uv run pytest tests/test_phase14_test_layers.py -q`; expected failure because the markers/targets are absent.**
- [ ] **Step 3: Add marker declarations and fast public commands; keep Compose E2E outside `make check`.**
- [ ] **Step 4: Run the focused test and marker collection; expected pass.**
- [ ] **Step 5: Run the existing public suite to detect marker-selection regressions.**

### Task 2: Add unit contract hardening

**Files:**
- Create: `tests/test_phase14_unit_contracts.py`
- Modify only existing source if a genuine contract gap is exposed by a failing test.

**Interfaces:**
- Tests the existing public APIs for normalization, article parsing, chunk models, fusion, grounding/citation, answerability, and Pydantic schemas.
- No duplicate broad examples where existing tests already prove the invariant.

- [ ] **Step 1: Write failing/invariant-focused tests for normalization policy behavior/idempotence, malformed article references, source spans, frozen fusion weights/ties/sparse-only/dense-only, citation metadata preservation, pre-generation abstention, and strict schema references.**
- [ ] **Step 2: Run the focused module and confirm failures identify missing Phase 14 assertions or genuine defects.**
- [ ] **Step 3: Add only minimal production fixes for genuine defects, preserving Phase 7–13 contracts.**
- [ ] **Step 4: Run the focused module and affected existing tests until green.**

### Task 3: Create the public synthetic PDF and reusable integration fixtures

**Files:**
- Create: `tests/fixtures/phase14/synthetic_appeals_regulation.pdf`
- Create: `tests/fixtures/phase14/README.md`
- Create: `tests/fixtures/phase14/corrupt.pdf`
- Create: `tests/integration/conftest.py`

**Interfaces:**
- Fixture identity is `phase14-synthetic-appeals-regulation`, with fictional English and Arabic provisions only.
- The PDF is machine-readable, text-based, tiny, and committed; corrupt input must fail closed.

- [ ] **Step 1: Add fixture tests that read the PDF through the selected lightweight parser boundary and assert identity/content; assert corrupt input fails.**
- [ ] **Step 2: Run the fixture tests to verify the failure before adding the fixture/parser adapter.**
- [ ] **Step 3: Generate the final PDF using temporary tooling only, add only the resulting bytes, and add a test-only parser helper under `tests/integration`.**
- [ ] **Step 4: Run fixture tests and verify no raw/private paths are referenced.**

### Task 4: Add deterministic subsystem integration coverage

**Files:**
- Create: `tests/integration/test_pdf_to_chunks.py`
- Create: `tests/integration/test_chunks_to_numpy.py`
- Create: `tests/integration/test_query_to_answer.py`
- Create: `tests/integration/test_citations.py`

**Interfaces:**
- Reuses real `parse_article_label`, normalization policies, chunk builders, `NumpyExactIndex`, `BM25Index`, `fuse_ranked_hits`, `HybridServingRetriever`, `ServingAnswerer`, `CanonicalCorpusResolver`, and citation verification.
- Uses deterministic lightweight embedding, reranker, generator, and six-to-ten chunk synthetic corpus.

- [ ] **Step 1: Write failing pipeline assertions for PDF identity, article units, exact text, valid spans, deterministic chunk IDs, provenance, and source-unit boundaries.**
- [ ] **Step 2: Run and confirm the integration failures are scoped to missing fixture wiring.**
- [ ] **Step 3: Implement test-only fixture builders and adapters; do not add hidden production branches.**
- [ ] **Step 4: Add index correspondence, dimension/duplicate rejection, frozen 50/50/1.0/0.25/60/20/8/raw-logit assertions, grounded answer, abstention, and four independent citation mutations.**
- [ ] **Step 5: Run all `tests/integration` tests and existing API/grounding/retrieval tests.**

### Task 5: Add locked public regression suite

**Files:**
- Create: `data/regression/phase14_cases.json`
- Create: `data/regression/README.md`
- Create: `data/manifests/testing/phase14_regression_lock.json`
- Create: `tests/regression/conftest.py`
- Create: `tests/regression/test_phase14_regression.py`
- Create: `tests/regression/test_phase14_lock.py`

**Interfaces:**
- Approximately 20 synthetic cases cover Arabic/English, article/deadline/authority/entity, multi-article, insufficient evidence, abstention, ties, and grounded answers.
- Cases freeze observable outcomes (top-K inclusion, answer/abstain, expected citation identity), never floating scores unless unavoidable.
- Lock records normalization, parsing, chunking, dense/reranker identities/revisions, fusion, generation, answerability, fixture hash, and case hash.

- [ ] **Step 1: Write tests that require the case schema, no HOLDOUT labels/paths, lock categories, stable hashes, and current-config equality.**
- [ ] **Step 2: Run them and confirm the lock/case files are absent or invalid.**
- [ ] **Step 3: Add the synthetic cases, lock metadata, and a read-only hash/config verifier; never rewrite baselines automatically.**
- [ ] **Step 4: Run the hermetic regression suite and verify deterministic repeated output.**

### Task 6: Add optional model-artifact regression and private smoke markers

**Files:**
- Create: `tests/model_regression/test_bge_cache_only.py`
- Create: `tests/e2e/test_private_smoke.py`
- Modify: `Makefile`

**Interfaces:**
- Model tests use only existing local BGE-M3 and BGE reranker caches; no download flag or network fallback.
- Private smoke is marked `private_artifact` and skips with an explicit reason when Phase 12 artifacts/models/Ollama are unavailable.

- [ ] **Step 1: Write cache-discovery and skip-reason tests.**
- [ ] **Step 2: Run them in the current environment; expected explicit skip if caches are unavailable.**
- [ ] **Step 3: Add the real-cache path and private smoke composition using existing `create_app()`/serving routes only.**
- [ ] **Step 4: Run the model/private targets and record result without treating skips as public failures.**

### Task 7: Add test-only Docker Compose E2E

**Files:**
- Create: `docker-compose.e2e.yml`
- Create: `docker/e2e/Dockerfile`
- Create: `docker/e2e/requirements.txt`
- Create: `tests/e2e/run_public_e2e.py`
- Create: `tests/e2e/test_streamlit_display.py`
- Modify: `Makefile`

**Interfaces:**
- Official Python 3.12 slim image, no forced platform, no privileged ports, no external services.
- Compose service healthchecks, deterministic startup, `--abort-on-container-exit`, `--exit-code-from e2e`, and unconditional `down -v --remove-orphans`.
- Runner proves ingest, canonical/chunk/index creation, health, Arabic search Article 12, grounded answer, exact citation, Streamlit Ask display, abstention, and clean exit.

- [ ] **Step 1: Write the runner assertions and a Compose smoke test that fails when the service/image is absent.**
- [ ] **Step 2: Run the focused test and verify the expected missing-stack failure.**
- [ ] **Step 3: Add the minimal test image and Compose wiring using dependency injection and deterministic adapters.**
- [ ] **Step 4: Run the exact Compose command locally; inspect architecture and cleanup.**
- [ ] **Step 5: Keep this target out of ordinary `make check` and preserve logs on failure.**

### Task 8: Document and report the hardening

**Files:**
- Create: `docs/testing.md`
- Create: `docs/reports/phase-14-testing-report.md`
- Modify: `README.md`

**Interfaces:**
- Documents each layer, commands, public/hermetic vs cache/private requirements, baseline-update procedure, Apple Silicon support, Compose usage/cleanup, and no-Qdrant rationale.
- README only surfaces `make test-regression`, `make test-e2e`, and the guide.

- [ ] **Step 1: Add documentation tests for required commands, matrix headings, no-Qdrant statement, and baseline-update procedure.**
- [ ] **Step 2: Run them to confirm missing docs fail.**
- [ ] **Step 3: Write the concise guide and handoff report with measured results only; do not claim unavailable CI/local Docker results.**
- [ ] **Step 4: Run docs tests and link checks.**

### Task 9: Verify quality and handoff

**Files:**
- Modify: `docs/reports/phase-14-testing-report.md` only with measured final results.

- [ ] **Step 1: Run `make check`.**
- [ ] **Step 2: Run `make test-integration`, `make test-regression`, and `make test-e2e` with fresh output.**
- [ ] **Step 3: Run model/private targets and record explicit skip/results.**
- [ ] **Step 4: Inspect diff for HOLDOUT/private data/secrets/machine paths/Qdrant/platform emulation.**
- [ ] **Step 5: Push the branch and open, but do not merge, the requested PR only after local verification; capture PR metadata and CI state.**
