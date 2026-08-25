# Phase 11B Stage B2 Prompt Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the final bounded Phase-11 DEV-only Stage B2 prompt experiment with synthetic exact-extraction demonstrations and an isolated clean checkpoint/result namespace.

**Architecture:** Preserve the existing provider schema, server-side field-local validation, candidate registry, and resumable runtime. Version only the prompt/config selection, route B2 to a separate private namespace, and retain B1 as the default existing path.

**Tech Stack:** Python 3.12, Pydantic contracts, argparse CLI, `uv`, pytest focused tests, Ruff, targeted Pyright.

**Spec:** User-approved Phase-11 Stage B2 prompt-only correction request in the current task.

## Global Constraints

- DEV only; HOLDOUT must remain inaccessible.
- No Qwen/Ollama inference, model download, semantic evaluation, or annotation/candidate changes in this task.
- Preserve model identity, schema hash, candidates-v3, validator behavior, runtime architecture, token budget, and automatic retries=0.
- Do not overwrite B0, B1 replay, B1 clean, or B1 evaluation artifacts.
- Tracked metadata must remain text-free; canonical text and raw responses remain private.
- Stage B2 is the final DEV prompt experiment; no Stage B3 or further tuning branch is added.

---

### Task 1: Add failing B2 prompt and namespace tests

**Files:**
- Modify: `tests/test_extraction_stage_b1.py`
- Modify: `tests/test_extraction_hybrid_runtime.py`

**Interfaces:**
- Tests consume the new B2 prompt version/hash, provider prompt selection, CLI stage selection, and B2 namespace constants.
- Tests must prove B1 defaults remain unchanged and B2 uses a separate clean namespace without invoking a provider during preflight.

- [ ] Add focused assertions for the four synthetic examples, regulated-entity guidance, complete-action guidance, multi-rule guidance, optional typed candidate references, unchanged schema hash, and synthetic-only example text.
- [ ] Add focused assertions for B2 result/checkpoint/config paths, 80 DEV inputs, zero initial completions, and HOLDOUT rejection.
- [ ] Run the focused tests and observe the expected failures before implementation.

### Task 2: Implement the versioned B2 prompt

**Files:**
- Modify: `src/kawaneen/extraction/hybrid_prompt.py`
- Modify: `src/kawaneen/extraction/provider.py`

**Interfaces:**
- Preserve `render_hybrid_prompt(text, registry)` as the B1-compatible default.
- Add an explicit B2 template selector and `hybrid_prompt_hash(template_version=...)` without changing `hybrid_schema_hash()`.
- Allow `OllamaExtractionProvider` to receive the selected prompt template version while retaining its current default.

- [ ] Keep the existing B1 instruction string and hash behavior unchanged.
- [ ] Add only the approved B2 additions: four short synthetic examples, explicit regulated-entity definition, nonempty extraction guidance, complete-action/bare-trigger rejection, distinct-rule guidance, and optional candidate-reference abstention.
- [ ] Ensure the B2 prompt contains typed allowlists and the canonical input only at render time; do not persist prompt/source text in tracked metadata.

### Task 3: Wire the isolated B2 clean run and preflight

**Files:**
- Modify: `src/kawaneen/extraction/orchestration.py`
- Modify: `src/kawaneen/extraction/hybrid_runtime.py`
- Modify: `src/kawaneen/cli.py`

**Interfaces:**
- Extend `run_hybrid_split` with an explicit B2 stage selector while preserving the current default B1-clean behavior.
- Add `--stage b2` and a no-inference `--preflight-only` path.
- Add B2 private result/checkpoint roots and text-free tracked configuration metadata.
- Reuse `run_hybrid_records` and timeout-only retry semantics; do not alter validation logic.

- [ ] Route B2 to `hybrid-qwen-v1-stage-b2-clean/dev` result/checkpoint namespaces and the B2 prompt hash.
- [ ] Require the existing locked DEV fingerprints and provenance checks.
- [ ] Make preflight verify 80 DEV inputs, zero completed B2 checkpoints, pending=80, static model/config identity, and no HOLDOUT path.
- [ ] Keep result/checkpoint extraction schema fields unchanged; store B2 stage identity only in experiment/outer metadata.

### Task 4: Run focused verification and privacy checks

**Files:**
- Verify: changed Python files and B2 metadata paths.

- [ ] Run focused prompt/runtime/orchestration tests only.
- [ ] Run Ruff on changed Python files and targeted Pyright.
- [ ] Run the B2 preflight without a provider call.
- [ ] Verify tracked metadata is text-free, private B2 paths are ignored/untracked, B1 paths are untouched, and no HOLDOUT access occurred.
- [ ] Compute B2 prompt/config hashes and report the exact manual inference command without executing it.
