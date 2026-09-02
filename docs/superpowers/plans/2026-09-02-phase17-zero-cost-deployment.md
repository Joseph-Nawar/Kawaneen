# Phase 17 — Zero-Cost Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible full-local Docker Compose profile and a visibly reduced, synthetic, retrieval-first public-demo bundle without changing frozen Phase 3–16 behavior.

**Architecture:** Keep the current FastAPI/Streamlit factories and frozen serving contracts. Add a small exact Qdrant adapter selected only by settings in full Compose, while the default remains NumPy. Compose mounts private artifacts read-only, uses named service volumes, and bootstraps Qdrant/Ollama with idempotent jobs. The public demo is an injected container built from synthetic chunks and precomputed E5-small vectors, with no LLM, Qdrant, MLflow, or private-artifact access.

**Tech Stack:** Python 3.11/3.12, uv, Pydantic Settings, FastAPI, Streamlit, NumPy, BM25/RRF, optional qdrant-client, Docker Compose, Mermaid, pytest, Ruff, Pyright.

**Spec:** User-provided “Implement Phase 17 — Zero-Cost Deployment for Kawaneen” specification in the conversation.

## Global Constraints

- The authoritative baseline is `origin/main` SHA `9203f7fc45a239ea056ea40c83a97d65ce12805b`.
- Work only on isolated branch `phase17/zero-cost-deployment`; create one DRAFT PR and never merge it.
- Full-local canonical command is `docker compose up`; `docker-compose.e2e.yml` remains test-only.
- Full-local defaults outside Compose remain NumPy, localhost Ollama, and existing Phase 8/10/11/16 semantics.
- Full Compose uses exact Qdrant search, frozen private artifacts through a read-only bind mount, and `KAWANEEN_*` settings.
- Public demo uses synthetic `KAWANEEN_DEMO` content, multilingual-e5-small revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, precomputed vectors, no LLM generation, and strict limits.
- Do not access or modify HOLDOUT, frozen result/configuration files, private corpus text, credentials, or paid services.
- New evaluation/qualification artifacts are additive and labelled `PHASE17_DEV`; no full or demo claim represents the full frozen production system.
- Public CI remains hermetic and must not require private artifacts, model downloads, network, live Qdrant, Ollama, or MLflow.

---

### Task 1: Settings and configurable Ollama endpoint

**Files:**
- Modify: `src/kawaneen/core/config.py`
- Modify: `src/kawaneen/api/composition.py`
- Modify: `src/kawaneen/generation/ollama.py`
- Modify: `src/kawaneen/extraction/provider.py`
- Test: `tests/test_config.py`, `tests/test_api_composition.py`, `tests/test_generation_adapters.py`, `tests/test_extraction_hybrid_runtime.py`

**Interfaces:**
- `Settings.ollama_url`, `Settings.qdrant_url`, and `Settings.dense_index_backend` are environment-backed settings.
- `OllamaGenerator` and `OllamaExtractionProvider` accept HTTP service hostnames while preserving localhost defaults.
- Stage-D and hybrid composition derive `/api/generate` from `Settings.ollama_url` without changing prompts, models, decoding, retries, timeout, verifier, or policy.

- [ ] Write tests proving defaults, env overrides, and endpoint propagation to Stage-D/hybrid providers.
- [ ] Run the focused tests and observe failure because settings and composition still hardcode localhost.
- [ ] Add the three settings and replace only the two composition endpoint literals; broaden endpoint validation to allow a valid HTTP service hostname.
- [ ] Run focused tests and the existing generation/extraction suites.
- [ ] Commit `feat: make local Ollama endpoint configurable`.

### Task 2: Exact Qdrant adapter and private-artifact bootstrap

**Files:**
- Create: `src/kawaneen/retrieval/qdrant_index.py`
- Create: `src/kawaneen/retrieval/qdrant_bootstrap.py`
- Modify: `src/kawaneen/retrieval/vector_index.py`
- Modify: `src/kawaneen/api/composition.py`
- Modify: `pyproject.toml`, `uv.lock`
- Test: `tests/test_retrieval_qdrant.py`, `tests/test_qdrant_bootstrap.py`

**Interfaces:**
- `QdrantExactIndex.build(client, collection_name, vectors, chunk_ids, corpus_hash, model_revision)` validates normalized float32 vectors and IDs.
- `QdrantExactIndex.search(query, top_k)` returns `tuple[ScoredChunk, ...]`, sends `exact=True`, validates normalized query dimensions, and sorts by `(-score, chunk_id)`.
- `seed_qdrant_collection(...)` validates `chunks.jsonl`, `vectors.npy`, `ids.json`, frozen corpus identity, dimension, count, and safely reuses/rebuilds only the Phase-17-owned collection.

- [ ] Write mock-client tests for exact search payload, query validation, result conversion/sorting, malformed hits, and bootstrap idempotency/mismatch/unrelated-collection safety.
- [ ] Run the focused tests and observe failure because the adapter/bootstrap do not exist.
- [ ] Add the minimal qdrant-client-backed adapter and bootstrap helpers; preserve `NumpyExactIndex` and use the Qdrant adapter only when `dense_index_backend == "qdrant"`.
- [ ] Run focused tests and existing vector-index tests with synthetic vectors only.
- [ ] Add a text-free parity runner/CLI that selects stable DEV IDs before querying and writes `data/evaluation/phase17_qdrant_parity.json` with `PHASE17_DEV`, `holdout_used=false`, top-k 50, identities, exact mode, mismatch count, and score tolerance.
- [ ] Commit `feat: add exact Qdrant serving backend`.

### Task 3: Full-local Compose and runtime images

**Files:**
- Create: `compose.yaml`
- Create: `docker/full/Dockerfile`, `docker/ui/Dockerfile`, `docker/requirements.txt`
- Create: `docker/qdrant-init.py`, `docker/ollama-init.py`
- Modify: `.dockerignore`, `.gitignore`, `Makefile`
- Test: `tests/test_phase17_deployment.py`

**Interfaces:**
- `docker compose up` defines long-running `kawaneen-api`, `kawaneen-ui`, `qdrant`, `mlflow`, `ollama` and short-lived `qdrant-init`, `ollama-init` jobs.
- Host bindings are loopback-only on ports 8000, 8501, 6333, 5000, and 11434; internal URLs use service names.
- `KAWANEEN_HOST_ARTIFACTS_DIR` defaults to `./artifacts` and is mounted read-only as `/app/artifacts`; mutable service data uses named volumes.
- Health checks and `service_healthy`/`service_completed_successfully` dependencies replace arbitrary sleeps.

- [ ] Write structural tests for service names, ports, mounts, healthchecks, dependency conditions, image contexts, and exclusion of private artifacts from build contexts.
- [ ] Run them and observe failure because the root Compose profile and runtime definitions are absent.
- [ ] Add minimal Python 3.12 runtime images, loopback-safe Compose, and idempotent initialization command wiring.
- [ ] Run the structural tests and `docker compose config`.
- [ ] Commit `feat: add full local Compose deployment`.

### Task 4: Synthetic public demo corpus and retrieval-first container

**Files:**
- Create: `data/demo/corpus/chunks.jsonl`, `data/demo/ids.json`, `data/demo/vectors.npy`, `data/demo/manifest.json`
- Create: `src/kawaneen/demo/corpus.py`, `src/kawaneen/demo/retrieval.py`, `src/kawaneen/demo/runtime.py`, `src/kawaneen/demo/limits.py`
- Modify: `src/kawaneen/api/contracts.py`, `src/kawaneen/api/app.py`, `src/kawaneen/api/runtime.py`, `src/kawaneen/api/routers.py`, `src/kawaneen/ui/config.py`, `src/kawaneen/ui/app.py`, relevant UI pages
- Test: `tests/test_phase17_demo.py`, `tests/test_phase17_limits.py`, `tests/test_phase17_ui.py`

**Interfaces:**
- `load_demo_corpus()` validates synthetic status, corpus/vector hashes, unique IDs, vector dimension/count, E5 formatting/normalization, and rejects private-source identifiers.
- `create_demo_container()` injects BM25 + E5-small/NumPy exact + RRF and optional qualified reranker over top four only; `create_demo_app()` reuses `create_app(container_factory=...)`.
- Demo answers are deterministic exact evidence snippets or explicit insufficient-evidence abstention; there is no generator capability and no Ollama path.
- Demo safeguards enforce query <=500 chars, extraction <=8000 chars, evidence <=5, one concurrent request, fixed-window ~30/minute, and ~15–20 second API timeouts.

- [ ] Write tests for corpus manifest integrity, deterministic sparse/dense/RRF retrieval, exact evidence provenance, no-generator capability, caps, rejection/error contracts, rate/concurrency behavior, and public-demo UI flag/banner/upload behavior.
- [ ] Run the focused tests and observe failure because the demo package and mode do not exist.
- [ ] Generate 60–100 fictional Arabic chunks with explicit KAWANEEN_DEMO disclaimers and deterministic 384-dimensional normalized vectors using the locked E5 contract (or a deterministic test-compatible generator if the model cache is unavailable; record the actual creation command in the manifest).
- [ ] Add the demo loader/retriever/container, API safeguard middleware/dependencies, explicit public-demo flag/banner, and disabled upload path; keep full/default mode unchanged.
- [ ] Run focused tests, existing API/UI suites, and verify no demo module imports Ollama/Qdrant/MLflow.
- [ ] Commit `feat: add synthetic retrieval-first public demo`.

### Task 5: Hugging Face Space export and qualification commands

**Files:**
- Create: `deploy/hf-space/Dockerfile`, `deploy/hf-space/entrypoint.sh`, `deploy/hf-space/README.template.md`, `deploy/hf-space/export.py`
- Create: `scripts/phase17_space_bundle.py`, `scripts/phase17_demo_qualify.py`, `data/evaluation/phase17_demo_qualification.json`
- Modify: `Makefile`, `.gitignore`
- Test: `tests/test_phase17_space_bundle.py`, `tests/test_phase17_qualification.py`

**Interfaces:**
- `make phase17-space-bundle` deterministically creates an ignored export containing only selected source, lock/dependencies, demo data, Space runtime files, README, and a hash manifest.
- The one-container entrypoint serves FastAPI on loopback `:8000` and Streamlit on `0.0.0.0:7860`, with no Qdrant/MLflow/Ollama process or private-artifact access.
- `make phase17-demo-qualify` records platform/architecture, limits, image metadata, timing/memory/error aggregates, reranker decision, local qualification status, and `NOT_PUBLISHED_USER_APPROVAL_REQUIRED`.

- [ ] Write export tests that assert allowlisted files/hashes and absence of `.git`, private artifacts, `.env`, credentials, full corpus, Ollama/Qwen/MLflow/Qdrant state.
- [ ] Run them and observe failure because the export command and bundle do not exist.
- [ ] Add the allowlisted exporter, single-container entrypoint, template README, and deterministic native qualification runner with one reranker attempt followed by one disabled fallback only on resource/reliability failure.
- [ ] Run focused tests and a local bundle build/smoke where Docker/model availability permits; record blocked measurements honestly rather than fabricating them.
- [ ] Commit `feat: add non-published Hugging Face demo bundle`.

### Task 6: Documentation, diagram, demo materials, and final audits

**Files:**
- Create: `docs/deployment/full-local.md`, `docs/deployment/public-demo.md`, `docs/deployment/api-examples.md`
- Create: `docs/demo/three-minute-script.md`, `docs/demo/shot-list.md`, `docs/architecture/phase17-deployment.mmd`
- Modify: `README.md`
- Create/add only if genuinely captured: `docs/assets/phase17/*.png`

**Interfaces:**
- Documentation names the project Kawaneen, distinguishes full-local from public demo, uses actual Phase 12 request contracts, and states publication remains approval-gated.
- Mermaid diagram shows both profiles and every required service/data-flow annotation.
- Demo script has exact narration, screens, Arabic queries/actions, visible results, and timestamps totaling exactly three minutes.

- [ ] Write documentation tests for required links, names, disclaimers, actual curl paths/JSON fields, and diagram nodes/annotations.
- [ ] Run them and observe failure for missing Phase 17 docs.
- [ ] Add concise recruiter-facing docs, diagram, script, shot list, and screenshot capture guide/status when GUI capture is unavailable; do not fabricate screenshots/video.
- [ ] Run docs tests and inspect rendered/linked content.
- [ ] Commit `docs: document Phase 17 deployment profiles`.

### Task 7: Full verification, frozen-result audit, and draft PR

**Files:**
- Modify only additive Phase 17 aggregate/report files if verification produced them.

- [ ] Run `uv run ruff format --check .`, `uv run ruff check .`, `uv run pyright`, `make check`, focused Phase 17 tests, `docker compose config`, and `make test-e2e`.
- [ ] Run the real full-local Compose smoke only with an existing external private artifact root mounted read-only; verify health, models, search, answer, deterministic extraction, Qdrant, and MLflow trace hierarchy without recording source text.
- [ ] Run Qdrant parity on DEV only; never access HOLDOUT.
- [ ] Build/smoke the Space bundle and `linux/amd64` image if feasible; label emulation and Apple Silicon measurements correctly.
- [ ] Verify `git ls-files artifacts/private` is empty, inspect `.env`/secret/private-path exposure, inspect Docker contexts, and confirm frozen Phase 8–16 files are unchanged.
- [ ] Run `git diff --check`, review the complete diff, create/update exactly one DRAFT PR against `main`, and never merge it.

## Self-review coverage

The tasks cover configurable endpoints, exact Qdrant/bootstrap/parity, full Compose, private-artifact boundaries, synthetic corpus, E5-small retrieval-first demo, safeguards, UI banner/mode, Space bundle, local qualification, docs/API examples/diagram, benchmark source-of-truth guidance, three-minute materials, security/frozen-result audits, and required verification. Video/screenshots remain explicitly conditional on genuine capture availability.
