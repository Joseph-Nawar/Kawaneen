# Phase 17 Merge-Readiness Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the existing Phase 17 draft PR so public Compose CI, real DEV Qdrant parity, demo semantics, self-contained Space readiness, and constrained qualification are truthful and reproducible.

**Architecture:** Keep the existing Phase 17 deployment design. Reuse the frozen Phase 8 DEV retrieval inputs and BGE-M3 encoder for parity, keep the public demo synthetic and deterministic, preload locked demo models during container startup, and qualify only from a constrained Docker run with measured health and resource data.

**Tech Stack:** Python 3.12, uv, pytest, FastAPI, Streamlit, Docker Compose, Qdrant, Hugging Face Hub cache, sentence-transformers, GitHub Actions.

**Spec:** User-provided Phase 17 merge-readiness repair brief, 2026-09-03.

## Global Constraints

- Do not redesign Phase 17 or change frozen Phase 8–16 model/metric behavior.
- Do not access HOLDOUT, model-shop, publish a Space, merge, or add payment actions.
- Do not persist private query text or query vectors; tracked artifacts remain text-free.
- Preserve exclusions for `artifacts/private`, `.env`, private data, and observability runtime state.
- Run all local commands with `uv` and finish with the requested CI, deployment, and governance audits.

### Task 1: Restore public Docker Compose E2E build

**Files:**
- Modify: `.dockerignore`
- Test: existing `tests`/Docker Compose E2E checks if a regression assertion is needed

**Interfaces:** The existing `docker/e2e/Dockerfile` must be able to copy `tests/e2e`, `tests/fixtures/phase14`, and `tests/phase14_support.py` while private/runtime exclusions remain active.

- [ ] Remove only the blanket `tests` exclusion or replace it with a Dockerfile-specific ignore that keeps the E2E inputs available.
- [ ] Run `make test-e2e` and confirm the existing public Compose E2E job passes.

### Task 2: Reuse frozen DEV inputs for real Qdrant parity

**Files:**
- Modify: `scripts/phase17_qdrant_parity.py`
- Modify: `src/kawaneen/retrieval/qdrant_parity.py` only if the aggregate contract needs the existing loader
- Create or modify: `data/evaluation/phase17_qdrant_parity.json`
- Test: `tests/test_phase17_parity.py`

**Interfaces:** Select 20 DEV query IDs deterministically before comparison; encode each query once with frozen BGE-M3; pass one normalized float32 vector to both `NumpyExactIndex` and `QdrantExactIndex`; emit only aggregate metadata.

- [ ] Add a failing regression test proving the runner uses the existing Phase 8/15 DEV asset and refuses HOLDOUT.
- [ ] Run the test red, then implement the loader and fixed-ID selection.
- [ ] Run the real gate at `top_k=50`; require identical IDs/order, max error `<=1e-5`, and zero mismatches.
- [ ] Persist the text-free aggregate with schema, provenance, selection hash, corpus/model identity, exact-Qdrant flag, and PASS/FAIL.

### Task 3: Correct partial reranker ordering and score metadata

**Files:**
- Modify: `src/kawaneen/demo/retrieval.py`
- Modify: API contracts/formatting only if mixed score types need representation
- Test: `tests/test_phase17_demo.py`

**Interfaces:** Fuse up to 8 candidates; rerank only the first 4; sort only that head by `-reranker_logit`, prior fused rank, and `chunk_id`; append the untouched tail in fused order; never compare reranker logits to RRF scores.

- [ ] Add a failing test with negative reranker logits and positive RRF scores proving the tail cannot jump ahead.
- [ ] Implement head-only sorting and truthful per-item/summary score types.
- [ ] Run the focused demo test and the full Phase 17 suite.

### Task 4: Add deterministic public-demo abstention

**Files:**
- Modify: `src/kawaneen/demo/retrieval.py`
- Modify: `src/kawaneen/demo/runtime.py`
- Modify: API contracts/routers only as required for the abstention payload
- Test: `tests/test_phase17_demo.py`

**Interfaces:** Filter sparse candidates with score `>0`; allow Ask only when selected evidence has positive lexical support; otherwise return `answerable=false` and `INSUFFICIENT_DEMO_EVIDENCE`; no generator is involved.

- [ ] Add failing tests for supported exact evidence, unrelated Arabic abstention, and no generator call.
- [ ] Implement the deterministic evidence gate without neural thresholds.
- [ ] Run the focused tests and verify the synthetic/not-real-law banner remains.

### Task 5: Make Space models self-contained and ready before health

**Files:**
- Modify: `deploy/hf-space/Dockerfile`
- Modify: `deploy/hf-space/entrypoint.sh`
- Modify: `src/kawaneen/demo/retrieval.py`
- Modify: `src/kawaneen/demo/runtime.py`
- Test: `tests/test_phase17_space_bundle.py`, `tests/test_phase17_demo.py`

**Interfaces:** The image downloads exact E5 and qualified BGE snapshots into the normal HF cache during build; runtime sets offline mode; `DemoRetriever.initialize()` loads the E5 adapter and optional reranker once; startup calls initialization before serving health.

- [ ] Add failing tests for adapter ownership and initialization-before-health behavior.
- [ ] Implement pinned model snapshot downloads and offline runtime configuration without Qwen/Ollama.
- [ ] Build linux/amd64 and run a no-host-cache, network-independent health/search/Ask smoke.

### Task 6: Replace host-only qualification with constrained-container qualification

**Files:**
- Modify: `src/kawaneen/deployment/qualification.py`
- Modify: `scripts/phase17_demo_qualify.py`
- Modify: `data/evaluation/phase17_demo_qualification.json`
- Test: `tests/test_phase17_qualification.py`

**Interfaces:** Use Docker CLI/subprocess with `--cpus=2 --memory=12g`, measure startup-to-health, idle/peak RSS, image identity/size, p50/p95 search/answer, fixed 20-query errors, and emit `HF_SPACE_RESOURCE_QUALIFIED` only when every required value and gate passes. Disable reranker once only if its resource/reliability gate fails.

- [ ] Add failing tests for missing required measurements and non-qualified decisions.
- [ ] Implement the constrained runner and text-free aggregate report.
- [ ] Run native constrained qualification, then a separate linux/amd64 compatibility smoke without using emulated latency as performance evidence.

### Task 7: Finish Compose verification and merge-readiness audit

**Files:**
- No additional production files unless a directly observed defect requires one.

**Interfaces:** Full Compose runs with the real external private artifact root read-only and verifies all service/init/health/model/trace/UI behaviors without tracking private content.

- [ ] Continue `docker compose up --build -d` until success or a genuine retryable failure is established.
- [ ] Verify Qdrant, Ollama frozen tag/digest, API search/answer/extraction, MLflow trace, and UI health.
- [ ] Run Ruff, Pyright, `make check`, focused Phase 17 tests, `make test-regression`, `make test-e2e`, `make phase17-verify`, parity, and audits.
- [ ] Commit repair changes, push the same branch/PR #13, wait for exact-head CI, and leave the PR draft/unmerged.
