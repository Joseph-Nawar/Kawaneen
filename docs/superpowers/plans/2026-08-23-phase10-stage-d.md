# Phase 10 Stage D — Answerability Policy and Direct-Only Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a qrel-free deterministic answerability/source-eligibility gate and isolated direct-only Qwen experiment namespace for Stage D.

**Architecture:** Stage D will use the frozen Phase-8/Phase-9 context universe and request-local QuoteRegistry, then apply a deterministic policy before generation. It will use a new direct-only Pydantic/provider schema, prompt version, fingerprints, readiness path, checkpoint/result/context/registry namespaces, and CLI generator identity. Stage C code paths and artifacts remain unchanged by defaulted compatibility parameters and separate Stage-D orchestration.

**Tech Stack:** Python 3.12, Pydantic v2, urllib/Ollama adapter, pytest, Ruff, Pyright, existing Phase-9 ContextPack/QuoteRegistry contracts.

**Spec:** User request: “Implement Phase 10 Stage D — deterministic answerability/source-eligibility policy plus direct-only generation.”

## Global Constraints

- Do not call Qwen/Ollama, Fanar, NLI, retrieval, holdout, full pytest/coverage, `make check`, commit, push, or Phase 11.
- Never use qrels, gold evidence, DEV labels, or evaluation populations in runtime policy decisions.
- Preserve Stage-C behavior and artifacts.
- Preserve model/digest, tokenizer, 60-second timeout, retries=0, temperature=0, sampling disabled, 3584 input cap, 512 output cap, max 3 claims, max 3 quote refs.
- Fail closed for future law, unverified currentness, unavailable authoritative source, missing dispositive section, unestablished case facts, and forum/source mismatch.

### Task 1: Add failing Stage-D policy and schema tests

**Files:**
- Modify: `tests/test_generation_policy.py`
- Modify: `tests/test_generation_stage_c.py` or create `tests/test_generation_stage_d.py`
- Modify: `tests/test_generation_cli.py`

**Interfaces:**
- Tests will define the required `evaluate_stage_d_policy`, `StageDGenerationPayload`, `stage_d_generation_payload_schema`, `STAGE_D_GENERATOR_NAME`, and CLI parser behavior.

- [ ] Add focused policy tests for future-law, enacted historical amendment, temporal currentness, official-primary-source eligibility, precedent-fact limitation, identified-case exception, missing/available dispositive material, forum/source mismatch, and no qrel dependency.
- [ ] Add direct-only schema tests for interpretation rejection, extra fields, claim/reference cardinality, and provider schema contents.
- [ ] Add CLI/status/readiness namespace tests and assert Stage-C paths remain distinct.
- [ ] Run the focused tests and confirm they fail because Stage-D interfaces are absent.

### Task 2: Implement deterministic answerability/source-eligibility policy

**Files:**
- Modify: `src/kawaneen/generation/contracts.py`
- Create: `src/kawaneen/generation/answerability.py`

**Interfaces:**
- `AbstentionReason` gains `FUTURE_LAW_UNKNOWABLE`, `AUTHORITATIVE_SOURCE_UNAVAILABLE`, `REQUIRED_CASE_SECTION_MISSING`, `CASE_FACTS_NOT_ESTABLISHED`, and `FORUM_OR_SOURCE_SCOPE_MISMATCH`.
- `evaluate_stage_d_policy(query: str, context: PolicyContext, *, source_registry: Mapping[str, SourceEligibility] | None = None, structural_roles: Mapping[str, str] | None = None, case_specific_evidence_available: bool = False) -> PolicyOutcome`.
- `SourceEligibility` is a frozen metadata contract loaded from governed `data/manifests/source_registry.csv`; no source names are hard-coded in policy rules.

- [ ] Implement governed source-registry loading using source role/type/authority/decision fields.
- [ ] Implement conservative intent-level classifiers for future law, temporal currentness, official/current text, unspecified case facts, dispositive outcome, and explicit forum/source scope.
- [ ] Call the existing jurisdiction/advice/currentness policy first and preserve its outcomes.
- [ ] Fail closed whenever required metadata or eligible evidence is unavailable.
- [ ] Keep qrels and evaluation labels out of all policy inputs.

### Task 3: Implement Stage-D direct-only contracts, prompt, and adapter

**Files:**
- Modify: `src/kawaneen/generation/contracts.py`
- Modify: `src/kawaneen/generation/prompt.py`
- Modify: `src/kawaneen/generation/quote_registry.py`
- Modify: `src/kawaneen/generation/ollama.py`
- Create: `src/kawaneen/generation/stage_d.py`

**Interfaces:**
- `StageDGenerationPayload` accepts only direct claims with `quote_refs`.
- `stage_d_generation_payload_schema()` returns the provider-side strict direct-only schema.
- `render_stage_d_generation_prompt()` and `stage_d_generation_version_hash()` are versioned independently.
- `stage_d_result_from_payload()` resolves only request-local QuoteRegistry references through the existing Phase-9 path.
- `OllamaGenerator(stage_d=True)` sends the Stage-D schema and parses the nested response with the shared extraction helper.

- [ ] Add schema and parser contracts before implementation changes.
- [ ] Add a Stage-D prompt that forbids interpretation/text/metadata and instructs direct-only compact output.
- [ ] Add Stage-D fingerprinting including policy/schema/prompt/registry/model/tokenizer/timeout/budget inputs.
- [ ] Extend the adapter with mutually exclusive Stage-D mode while preserving Stage-A/B/C defaults.

### Task 4: Parameterize Stage-D context caching and orchestration

**Files:**
- Modify: `src/kawaneen/generation/stage_c_context.py` with backward-compatible Stage-C defaults
- Modify: `src/kawaneen/generation/orchestration.py`
- Modify: `src/kawaneen/cli.py`
- Create: `src/kawaneen/generation/stage_d_context.py` only if parameterization cannot preserve a single responsibility

**Interfaces:**
- `qwen-ollama-stage-d` owns separate context, registry, readiness, checkpoint, and result roots.
- `generation_status`, `generation_readiness`, and `run_dev_generation` accept Stage D without falling through to Stage C.
- `stage_d_readiness()` builds/loads all 160 contexts and registries, applies policy without model access, and reports policy counts and eligible prompt-token summaries.

- [ ] Add Stage-D constants and namespace routing.
- [ ] Reuse frozen Phase-8/Phase-9 inputs only; pass no qrels into runtime policy or generation.
- [ ] Apply Stage-D policy after ContextPack/QuoteRegistry construction and before `OllamaGenerator.generate()`.
- [ ] Persist explicit pre-policy completion reasons and Stage-D fingerprints atomically.
- [ ] Ensure Stage-C paths/artifacts remain untouched.

### Task 5: Run offline readiness and focused verification

**Files:**
- Create: `data/evaluation/phase10_qwen_stage_d_readiness.json`
- Create: `artifacts/private/phase10_generation/readiness/qwen-ollama-stage-d/policy_audit.json`

- [ ] Run the Stage-D readiness path only; no Ollama/model calls.
- [ ] Confirm 160/160 contexts and registries, policy distribution, eligible prompt-token p50/p95/max, zero budget violations, and unresolved provenance count.
- [ ] Run focused Stage-D/policy/generation tests with coverage disabled if repository coverage floor prevents focused execution.
- [ ] Run Ruff, generation/grounding Pyright, deterministic fingerprint/provenance audit, and tracked/private-text audit.
- [ ] Report exact manual commands for Stage-D status and resumable DEV run.
