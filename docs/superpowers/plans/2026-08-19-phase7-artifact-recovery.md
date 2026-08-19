# Phase 7 Holdout Artifact Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the minimum sanitized per-query holdout replay data, recover the missing holdout analyses exactly once, and verify Phase 7 without changing frozen retrieval configuration.

**Architecture:** Extend only the holdout result persistence/reporting layer. The replay will reuse the frozen corpus, qrels, model locks, caches, policies, and ranking path, writing ID-bearing data only below the private artifact root and writing aggregate/hash/provenance data to tracked outputs.

**Tech Stack:** Python 3.12, pytest, NumPy, existing retrieval evaluators and checkpoint caches, Ruff, Pyright, pre-commit, uv.

**Spec:** User-provided “Finish Phase 7 with a bounded artifact-recovery and verification pass.”

## Global Constraints

- Do not alter frozen retrieval configuration, Phase 0–6 data, corpus, qrels, model settings, or normalization selections.
- Do not encode E5 or BGE corpus embeddings again.
- Perform exactly one deterministic holdout artifact-recovery replay with reason `artifact_recovery_after_instrumentation_defect`.
- No query, evidence, or document text may enter tracked artifacts.
- Do not run the full repository pytest/coverage suite until all code/report work is complete; run it once only.
- Do not commit, push, merge, or begin Phase 8.

### Task 1: Add sanitized holdout replay persistence

**Files:**
- Modify: `src/kawaneen/retrieval/orchestrator.py`
- Modify: `src/kawaneen/retrieval/artifacts.py`
- Test: `tests/test_retrieval_orchestration.py`
- Test: `tests/test_retrieval_artifacts.py`

- [ ] **Step 1: Write failing mocked tests** for persistence containing query ID, parent/base ID, retriever ID, ranked IDs/scores, qrel-derived indicators, evidence satisfaction, latency, and slice metadata; assert no text-bearing fields are written and that rankings/metrics are unchanged.
- [ ] **Step 2: Run only the new focused tests and confirm the expected failure.**
- [ ] **Step 3: Implement a private replay writer and invoke it for lexical and dense holdout methods without changing ranking or metric computation.**
- [ ] **Step 4: Run the focused tests and verify they pass.**

### Task 2: Add deterministic replay and report recovery

**Files:**
- Modify: `src/kawaneen/retrieval/orchestrator.py`
- Modify: `src/kawaneen/cli.py` only if an explicit recovery/replay operation is required by the existing CLI contract
- Test: `tests/test_retrieval_orchestration.py`

- [ ] **Step 1: Write failing tests** for frozen-gate validation, replay reason recording, exact aggregate comparison, and recovered complementarity/bootstrap/robustness/latency/unanswerable outputs.
- [ ] **Step 2: Run only those focused tests and confirm the expected failure.**
- [ ] **Step 3: Implement a bounded recovery path that refuses configuration changes and never encodes missing corpus embeddings.**
- [ ] **Step 4: Run focused retrieval tests and verify the replay assembly logic.**

### Task 3: Execute the single authorized recovery replay

- [ ] Validate the frozen selection, hashes, model revisions, and all cache manifests.
- [ ] Execute exactly one holdout replay with reason `artifact_recovery_after_instrumentation_defect`.
- [ ] Compare every required aggregate metric against the original holdout and stop on any mismatch.
- [ ] Assemble and hash the final report and manifest without using replay output for tuning.

### Task 4: Run bounded quality checks

- [ ] Run focused retrieval tests, Ruff, retrieval Pyright, deterministic/hash checks, private/tracked artifact audits, and pre-commit.
- [ ] Compare repository Pyright against clean Phase-6 commit `f81f30b` and record Phase-7-introduced errors.
- [ ] Run the repository-wide pytest + coverage gate exactly once after all work is complete.
- [ ] Stop and report if that final suite fails any gate.
