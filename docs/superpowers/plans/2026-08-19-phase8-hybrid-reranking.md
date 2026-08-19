# Phase 8 Hybrid Retrieval and Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase-8 hybrid retrieval/reranking infrastructure and run only cheap DEV RRF fusion experiments over frozen Phase-7 artifacts.

**Architecture:** A focused `kawaneen.retrieval.hybrid` package owns typed contracts, deterministic RRF fusion, explicit metadata eligibility, lazy BGE reranking, checkpointing, evaluation diagnostics, and text-free artifact writing. The experiment runner consumes the frozen Phase-7 selection, corpus manifest, private BM25 rankings, and existing BGE cache; it never changes Phase-7 inputs and refuses missing or mismatched artifacts.

**Tech Stack:** Python 3.11–3.12, dataclasses, NumPy, existing BM25/vector/evaluation primitives, TOML, atomic JSON writes, pytest, Ruff, Pyright.

**Spec:** User-provided “Phase 8 — Hybrid Retrieval and Reranking: Infrastructure + DEV Fusion” request.

## Global Constraints

- Do not run Phase 8 holdout, real reranker inference, corpus re-encoding, full pytest/coverage, or Phase 9.
- Reuse Phase-7 BM25/BGE settings, populations, qrels, metrics, slices, and holdout protections.
- Do not commit or push.
- Private rankings/checkpoints may contain IDs and scores; tracked artifacts remain text-free.
- Do not add weights, models, candidate depths, or rerankers beyond the fixed ladder.

---

### Task 1: Contracts, config, and deterministic fusion

**Files:**
- Create: `src/kawaneen/retrieval/hybrid/contracts.py`
- Create: `src/kawaneen/retrieval/hybrid/fusion.py`
- Create: `configs/retrieval/phase8_hybrid.toml`
- Test: `tests/test_retrieval_hybrid_fusion.py`

- [ ] Write failing tests for exact RRF arithmetic, provenance, weighted configurations, duplicate IDs, deterministic ties, top-20 truncation, and invalid configuration values.
- [ ] Run `uv run pytest tests/test_retrieval_hybrid_fusion.py --no-cov -q` and confirm the expected missing-package failure.
- [ ] Implement frozen source-hit, fused-candidate, and fusion-config contracts plus the fixed RRF implementation.
- [ ] Re-run the focused fusion tests until green.

### Task 2: Metadata eligibility and filtered retrieval

**Files:**
- Create: `src/kawaneen/retrieval/hybrid/metadata.py`
- Create: `src/kawaneen/retrieval/hybrid/filtered.py`
- Test: `tests/test_retrieval_hybrid_metadata.py`

- [ ] Write failing tests for all six fields, OR-within/AND-across semantics, inclusive dates, unknown metadata, invalid/empty filters, zero matches, masked BM25/dense ranking, and excluded IDs.
- [ ] Run the focused metadata tests and verify failure is caused by missing Phase-8 modules.
- [ ] Implement typed filters, explicit metadata indexes, text-free coverage reporting, and eligible-ID ranking helpers.
- [ ] Re-run the focused metadata tests until green.

### Task 3: Reranker adapter and resumable execution

**Files:**
- Create: `src/kawaneen/retrieval/hybrid/reranker.py`
- Create: `src/kawaneen/retrieval/hybrid/checkpoints.py`
- Test: `tests/test_retrieval_hybrid_reranker.py`

- [ ] Write failing mocked-scorer tests for original query/display text, passage-only truncation, ordering, tie-breaking, non-finite rejection, fingerprints, interruption resume, corrupt checkpoint recomputation, config invalidation, and manifest-only status.
- [ ] Run the focused reranker tests and verify the expected missing-module failure.
- [ ] Implement a lazy `BAAI/bge-reranker-v2-m3` adapter, pair diagnostics, score validation, atomic per-query checkpoints, resume validation, and status inspection.
- [ ] Re-run the focused reranker tests until green without loading a real model.

### Task 4: Pipeline and evaluation diagnostics

**Files:**
- Create: `src/kawaneen/retrieval/hybrid/pipeline.py`
- Create: `src/kawaneen/retrieval/hybrid/evaluation.py`
- Test: `tests/test_retrieval_hybrid_pipeline.py`

- [ ] Write failing tests for 50+50→RRF20→rerank10/serve8, provenance, CandidateRecall/CER, rescue/damage counts, contribution fractions, and bootstrap comparison guards.
- [ ] Run the focused pipeline tests and verify the expected missing-module failure.
- [ ] Implement retrieval/reranking composition and reuse existing Phase-7 metric/slice implementations.
- [ ] Re-run the focused pipeline tests until green.

### Task 5: Phase-8 orchestration, artifacts, CLI, and documentation

**Files:**
- Create: `src/kawaneen/retrieval/hybrid/artifacts.py`
- Create: `src/kawaneen/retrieval/hybrid/orchestration.py`
- Create: `src/kawaneen/retrieval/hybrid/__init__.py`
- Modify: `src/kawaneen/cli.py`
- Create: `data/manifests/retrieval/phase8_model_lock.json`
- Create: `data/manifests/retrieval/phase8_metadata_coverage.json`
- Create: `data/evaluation/phase8_dev_fusion_metrics.json`
- Create: `data/manifests/retrieval/phase8_dev_fusion_selection.json`
- Create: `docs/phases/phase-08-hybrid-reranking.md`
- Test: `tests/test_retrieval_hybrid_artifacts.py`

- [ ] Write failing tests for frozen Phase-7 SHA validation, text-free tracked payloads, protected holdout access, and CLI/status wiring.
- [ ] Implement artifact/manifests, cheap DEV fusion orchestration from existing private artifacts/cache, and `phase8-rerank-dev --resume --device cpu` plus manifest-only status.
- [ ] Run the cheap DEV fusion command only after focused tests pass; refuse missing/invalid caches and never call reranker scoring.
- [ ] Record the provisional selection and diagnostics without creating a final Phase-8 holdout manifest.
- [ ] Run the focused artifact/CLI tests until green.

### Task 6: Verification and handoff

**Files:**
- Modify: focused test files only if verification reveals defects.

- [ ] Run only Phase-8/retrieval focused pytest with `--no-cov`, Ruff, and retrieval-scoped Pyright.
- [ ] Confirm no holdout, reranker model, corpus encoding, or full repository pytest/coverage command was run.
- [ ] Inspect `git diff` and report changed files, DEV metrics, diagnostics, model contract, and exact manual reranking command without committing.
