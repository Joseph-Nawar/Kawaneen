# Phase 10 Stage A.1 Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase 10 real-generator readiness fail closed by wiring governed Saudi v1 jurisdiction scope, disabling extractive fallback, resolving immutable Qwen identities, persisting Ollama locks, and adding unexecuted resumable DEV plumbing.

**Architecture:** A text-free deployment contract will bind the active Phase-6 source release to `SA`, while policy normalization will reject conflicting and out-of-scope jurisdiction requests before any generator call. Generation orchestration will select only the locked Qwen Ollama adapter, budget frozen Phase-9 contexts with the locked tokenizer, atomically checkpoint private per-query results, and expose only status metadata through the CLI. The deterministic extractive generator remains available only as an explicitly named benchmark mode.

**Tech Stack:** Python 3.12 (`>=3.11,<3.13`), Pydantic contracts, `huggingface_hub` metadata API, local Ollama HTTP API via an injectable transport, existing Phase-8/Phase-9 readers and assembler, `uv`, focused pytest/Ruff/Pyright.

**Spec:** `docs/superpowers/specs/2026-08-21-phase10-generation-abstention-design.md` plus the Phase 10 Stage A.1 user request.

## Global Constraints

- Do not run Qwen, Fanar, NLI, retrieval, holdout, full pytest/coverage, `make check`, or any model download.
- Do not tune the extractive baseline or retrieval-sufficiency threshold; preserve `retrieval_score_gate=disabled`.
- No command may exceed 10 minutes; do not pull Ollama in this task.
- Do not commit or push.
- Phase-8 rankings, qrels, chunks, Phase-9 guarantees, and existing extractive DEV metrics remain immutable.
- Runtime generation may read persisted Phase-8/Phase-9 inputs but must not receive qrels.
- Source-bearing generation outputs remain under ignored `artifacts/private`; tracked artifacts contain only text-free metadata.

### Task 1: Governed Saudi jurisdiction contract

**Files:**
- Create: `data/manifests/generation/phase10_jurisdiction_scope.json`
- Modify: `src/kawaneen/generation/policy.py`
- Modify: `src/kawaneen/generation/contracts.py`
- Test: `tests/test_generation_policy.py`

**Interfaces:**
- `load_deployment_jurisdiction(path: Path = ...) -> JurisdictionScope` validates `SA`, `{SA}`, `single`, authoritative source path, and source-registry hash.
- `JurisdictionScope` carries normalized `active_jurisdiction`, `allowed_jurisdictions`, and `mode`, while preserving compatibility aliases for existing callers.
- `evaluate_pre_generation_policy` returns `JURISDICTION_MISMATCH` for explicit non-SA requests, `JURISDICTION_AMBIGUOUS` for multiple recognized jurisdiction markers/conflicting context metadata, and refuses unresolved scope when a required scope is configured.

- [ ] **Step 1: Write failing Arabic/English tests** for the default governed Saudi scope, Egyptian-vs-Saudi mismatch, Arabic Egypt mismatch, mixed Saudi/Egypt ambiguity, conflicting context source jurisdictions, and model-supplied jurisdiction being ignored by policy.
- [ ] **Step 2: Run only the new policy tests** and confirm the current default scope fails or allows the wrong cases.
- [ ] **Step 3: Add the text-free contract** citing `data/manifests/source_registry.csv` and the frozen Phase-6 source manifest; record `active_jurisdiction=SA`, `allowed_jurisdictions=[SA]`, `mode=single`.
- [ ] **Step 4: Implement normalized marker detection and fail-closed scope validation** before advice/currentness/context/generator handling.
- [ ] **Step 5: Run the focused policy tests** and confirm all jurisdiction cases pass.

### Task 2: Benchmark-only extractive and disabled retrieval gate

**Files:**
- Modify: `src/kawaneen/generation/extractive.py`
- Modify: `src/kawaneen/generation/generator.py`
- Modify: `src/kawaneen/generation/artifacts.py` or generation policy manifest writer
- Modify: `tests/test_generation_extractive.py`
- Create/modify: focused generator orchestration tests

**Interfaces:**
- `ExtractiveGenerator` remains unchanged in selection behavior and exposes `benchmark_only=True`.
- Production generator selection rejects extractive mode unless an explicit benchmark-only flag is supplied.
- Qwen failure, timeout, invalid JSON, invalid citation, or unsupported claim returns abstention and never invokes extractive fallback.

- [ ] **Step 1: Write failing tests** proving each Qwen failure path abstains and that the extractive generator cannot be selected by automatic fallback.
- [ ] **Step 2: Run those tests** and observe current fallback/selection behavior.
- [ ] **Step 3: Add the minimal benchmark-only marker and production selection guard** without changing lexical extraction.
- [ ] **Step 4: Run focused extractive/adapter tests** and verify the frozen baseline remains unchanged.

### Task 3: Immutable Qwen model and tokenizer metadata

**Files:**
- Modify: `src/kawaneen/generation/registry.py`
- Modify: `src/kawaneen/generation/tokenizer.py`
- Modify: `data/manifests/generation/phase10_model_generator_lock.json`
- Modify: `data/manifests/generation/phase10_tokenizer_budget.json`
- Test: `tests/test_generation_registry.py`, `tests/test_generation_budgeting.py`

**Interfaces:**
- `resolve_hf_revision_from_hub` returns a full 40-character SHA from Hub metadata only.
- `resolve_qwen_identity()` resolves model and matching tokenizer repository revision without loading weights.
- Lock validators reject short SHAs and mismatched model/tokenizer repository revisions where the contract requires matching.

- [ ] **Step 1: Write failing tests** for full SHA validation, Qwen model/tokenizer identity, shared revision recording, and no-weight/no-tokenizer-load behavior.
- [ ] **Step 2: Run the focused registry tests** with injected Hub metadata and verify missing locks fail.
- [ ] **Step 3: Implement metadata-only resolution and update the text-free lock manifests** with the observed full SHA.
- [ ] **Step 4: Run focused registry/budget tests** without loading generator weights or running tokenized DEV generation.

### Task 4: Ollama local digest lock and tag verification

**Files:**
- Modify: `src/kawaneen/generation/ollama.py`
- Modify: `src/kawaneen/generation/registry.py`
- Modify: `src/kawaneen/generation/checkpoints.py`
- Modify: `src/kawaneen/cli.py`
- Test: `tests/test_generation_adapters.py`, `tests/test_generation_checkpoints.py`

**Interfaces:**
- `OllamaTransport.get_json(endpoint, payload=None) -> object` supports mocked `/api/show` identity lookup.
- `inspect_ollama_model(endpoint, expected_tag, transport) -> OllamaModelIdentity` verifies the exact tag and returns the immutable digest.
- `write_local_model_lock(path, identity)` atomically persists text-free tag/digest metadata.
- `load_local_model_lock(path, expected_tag)` rejects tag or digest mismatch before generation.

- [ ] **Step 1: Write failing mocked-transport tests** for exact tag success, missing tag, digest mismatch, malformed identity, and lock persistence.
- [ ] **Step 2: Run the tests** and confirm the current flow only validates a caller-supplied digest.
- [ ] **Step 3: Implement local identity inspection, atomic lock persistence, and generation preflight validation.**
- [ ] **Step 4: Extend the CLI lock command** to query local Ollama only when manually invoked; do not execute it here.
- [ ] **Step 5: Run focused adapter/checkpoint/CLI tests** with no daemon.

### Task 5: Resumable Qwen DEV plumbing and status

**Files:**
- Create: `src/kawaneen/generation/orchestration.py`
- Modify: `src/kawaneen/generation/checkpoints.py`
- Modify: `src/kawaneen/generation/ollama.py`
- Modify: `src/kawaneen/generation/budgeting.py`
- Modify: `src/kawaneen/cli.py`
- Test: `tests/test_generation_orchestration.py`

**Interfaces:**
- `generation_status(generator_name: str) -> dict[str, object]` reads only text-free manifests/checkpoints and does not load model, tokenizer, corpus, source text, or qrels.
- `run_dev_generation(*, generator_name: str, resume: bool, ...) -> dict[str, object]` consumes frozen Phase-8 rankings and Phase-9 packs, assembles with the locked Qwen tokenizer, and writes private raw/final outputs plus atomic per-query checkpoints.
- `generation_fingerprint(...) -> str` includes query ID, Phase-8 SHA, Phase-9 policy/hash, ContextPack/input hash, model revision, Ollama digest, tokenizer revision, prompt-template hash, generation-policy hash, and decoding settings.

- [ ] **Step 1: Write failing tests** for model-free status, qrels-inaccessible runtime interfaces, per-query resume, corrupt checkpoint recomputation, and fingerprint invalidation when any lock/input changes.
- [ ] **Step 2: Run the focused orchestration tests** and confirm the commands/interfaces are absent.
- [ ] **Step 3: Implement status using only manifests and checkpoint metadata.**
- [ ] **Step 4: Implement run plumbing with dependency injection** for tokenizer, assembler, Ollama generator, policy, and private writer; never pass qrels into the runtime request.
- [ ] **Step 5: Implement atomic private source-bearing output and text-free tracked summaries.**
- [ ] **Step 6: Add CLI commands** equivalent to `generation status --generator qwen-ollama` and `generation run-dev --generator qwen-ollama --resume`.
- [ ] **Step 7: Run focused orchestration tests only; do not execute the command.**

### Task 6: Allowed verification and handoff

**Files:**
- Modify: `docs/reports/phase10-stage-a-handoff.md`
- Modify: `data/evaluation/phase10_retrieval_sufficiency_calibration.json`
- Modify: `data/manifests/generation/phase10_generation_policy.json`

- [ ] **Step 1: Run focused generation/grounding pytest with coverage disabled.**
- [ ] **Step 2: Run Ruff on changed Python files.**
- [ ] **Step 3: Run generation/grounding Pyright.**
- [ ] **Step 4: Run deterministic/hash and tracked/private text audits.**
- [ ] **Step 5: Hash every Stage-A.1 artifact and update the handoff with exact manual commands.**
- [ ] **Step 6: Report whether readiness gates pass; do not claim readiness if any lock, jurisdiction, or command gate remains unresolved.**

No commit or push is part of this plan because the user explicitly forbids both.
