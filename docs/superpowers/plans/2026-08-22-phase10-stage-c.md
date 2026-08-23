# Phase 10 Stage C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated Stage-C Qwen experiment that emits compact request-local quote references, resolves authoritative quotations server-side, and uses a 60-second Ollama timeout without changing Stage-A/B artifacts.

**Architecture:** Stage-C gets its own contracts, prompt version, context-pack/QuoteRegistry cache, checkpoint/result roots, and generator identity. The registry is derived from the final Phase-9 ContextPack in deterministic source order; model quote references are converted into Phase-9 CitationRequests by the server before unchanged verification and rendering. Stage-C readiness reuses the frozen Phase-8 rankings and production tokenizer but never constructs or calls a model.

**Tech Stack:** Python 3.12, Pydantic v2, urllib Ollama adapter, existing Phase-9 ContextAssembler/CitationVerifier, `uv`, focused pytest/Ruff/Pyright.

**Spec:** Phase-10 Stage-C compact server-resolved quotation references and runtime reliability request.

## Global Constraints

- Do not call Qwen/Ollama during implementation or readiness assembly.
- Preserve Stage-A/B artifacts, retrieval, Phase-9 verification, 3584 input cap, 512 output cap, three-claim limit, deterministic decoding, and safety policies.
- Stage-C uses `qwen-ollama-stage-c` and private namespaces distinct from all prior stages.
- No retries, streaming remains disabled, and Qwen failures abstain without extractive fallback.
- Tracked artifacts remain text-free; raw provider envelopes remain private.

### Task 1: Define Stage-C quote-reference contracts

**Files:** Modify `src/kawaneen/generation/contracts.py`; create `src/kawaneen/generation/quote_registry.py`; test `tests/test_generation_stage_c.py`.

- [ ] Add strict direct claims with only `mode` and `quote_refs`, and interpretation claims with only `mode`, `text`, and `quote_refs`; enforce one to three claims, one to three references, and `Q`-ID format.
- [ ] Derive the Stage-C provider schema from the Pydantic payload and reject Stage-B `quoted_text`, metadata, malformed IDs, excess claims, and excess references.
- [ ] Build a request-local registry from `ContextPack.evidence` in rendered source order, deduplicated by canonical unit ID, preserving exact display text, unit ID, evidence ID, chunk IDs, ranks, and source provenance.
- [ ] Test deterministic IDs, canonical text preservation, deduplication, local scope, unknown references, and fingerprint changes.

### Task 2: Resolve references through unchanged Phase-9 verification

**Files:** Modify `src/kawaneen/generation/postprocessing.py`; test `tests/test_generation_stage_c.py`.

- [ ] Convert Stage-C claims to the existing `ModelOutputClaim`/`CitationRequest` path using registry-resolved exact text and evidence IDs.
- [ ] Keep direct rendering server-controlled and keep interpretation claims unavailable without semantic support.
- [ ] Test verifier invocation, exact source rendering, unknown-reference abstention, and rejection of model-generated quotation/metadata fields.

### Task 3: Version Stage-C prompt, budgeting, and Ollama runtime

**Files:** Modify `src/kawaneen/generation/prompt.py`, `budgeting.py`, `ollama.py`; test `tests/test_generation_stage_c.py` and `tests/test_generation_adapters.py`.

- [ ] Render `[Qxxx]` labels beside canonical evidence and count them through the existing tokenizer budget.
- [ ] Add the Stage-C prompt/schema/policy hashes without changing Stage-B values.
- [ ] Set the Stage-C transport timeout to 60 seconds, preserving zero retries and non-streaming behavior; persist private native response telemetry and fail closed on transport errors.

### Task 4: Add isolated Stage-C orchestration and readiness

**Files:** Modify `src/kawaneen/generation/orchestration.py`, `cli.py`, and `checkpoints.py`; test `tests/test_generation_stage_c.py`, `tests/test_generation_orchestration.py`, and `tests/test_generation_cli.py`.

- [ ] Add Stage-C private context, QuoteRegistry, checkpoint, and result roots and a distinct experiment identity.
- [ ] Include registry policy/version, prompt/schema/policy hashes, model/digest, tokenizer, 60-second timeout, budget settings, and ordered provenance in fingerprints.
- [ ] Add status and resumable run commands; make status model/corpus-free, validate resume fingerprints, recompute corrupt entries, and never read Stage-B checkpoints as Stage-C results.
- [ ] Add a no-generation readiness path that builds all 160 contexts/registries and reports token, provenance, truncation, and post-hoc retention metrics.

### Task 5: Build readiness artifacts and verify

**Files:** Add text-free Stage-C readiness manifest under `data/evaluation/`; private outputs under `artifacts/private/phase10_generation/`.

- [ ] Run the bounded tokenizer/assembly readiness path only, never Ollama.
- [ ] Write only aggregate text-free tracked artifacts and audit hashes/private text.
- [ ] Run focused generation/grounding tests, Ruff, targeted generation/grounding Pyright, deterministic/hash audit, and tracked/private-text audit.

