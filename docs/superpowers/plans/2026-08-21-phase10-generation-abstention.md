# Phase 10 Stage A — Local Generation and Abstention Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, local-only generation and abstention infrastructure that accepts only claim-plus-citation proposals and remains grounded by the immutable Phase-9 ContextPack/verifier.

**Architecture:** A new `kawaneen.generation` package separates strict contracts, prompt rendering, generator adapters, tokenizer budgeting, safety policy, abstention, deferred semantic support, checkpoints, evaluation, and artifacts. All model-facing components are lazy and injectable; the extractive baseline is fully deterministic and the final answer is rendered only from verified Phase-9 claims.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, standard-library JSON/HTTP/hashlib/regex, existing Phase-9 grounding contracts, optional lazy Transformers/Ollama integrations, pytest, Ruff, and focused strict Pyright.

**Spec:** `docs/superpowers/specs/2026-08-21-phase10-generation-abstention-design.md`

## Global Constraints

- Consume Phases 8–9 as immutable dependencies; selection SHA is `a62cc772f2b71883355c7935da7e7b87ab4d22b3746553148b4f64ef20f28b0b`.
- No real Ollama/Transformers/NLI inference, weight downloads, Phase-10 holdout, Phase-8 retrieval, paid/external API, or Phase 11 work.
- Do not modify retrieval, grounding, qrels, chunks, normalization, holdout artifacts, or Phase-9 citation guarantees.
- Use exact evidence quotations and context-local evidence IDs only; metadata is resolved by Phase 9/server policy.
- Default generation settings are temperature `0`, sampling disabled, max new tokens `384`, max claims `3`, and no automatic retry.
- Total prompt/input cap is `3584` tokens, output reservation `384`, and safety margin `128`; evidence is allocated only at complete canonical-unit boundaries.
- Jurisdiction is server-controlled and unverified when no authoritative active single-jurisdiction scope exists.
- Run only focused tests, Ruff, and generation/grounding Pyright; do not run the full suite, coverage, or `make check`.
- Do not commit or push.

### Task 1: Define generation contracts and strict output parsing

**Files:**
- Create: `src/kawaneen/generation/contracts.py`
- Create: `src/kawaneen/generation/abstention.py`
- Create: `src/kawaneen/generation/__init__.py`
- Test: `tests/test_generation_contracts.py`

**Interfaces:**
- `GenerationDecision`, `AbstentionReason`, `GenerationSettings`, `ModelCandidate`, `ModelOutputCitation`, `ModelOutputClaim`, `ModelOutput`, `GenerationRequest`, `GenerationResult`.
- `parse_model_output(payload: str | bytes) -> ModelOutput`.
- `invalid_generation_result(detail: str | None = None) -> GenerationResult`.

- [ ] Write tests proving extra fields, answer/reasoning/metadata fields, empty IDs/quotes, answer with zero claims, abstain with claims, and more than three claims are rejected.
- [ ] Run `uv run pytest tests/test_generation_contracts.py -o addopts='' -v`; verify RED because the package is absent.
- [ ] Implement frozen `extra="forbid"` contracts with model-level decision/claim cardinality validation and fail-closed JSON parsing.
- [ ] Re-run the focused contract tests and verify GREEN.

### Task 2: Add versioned prompt rendering and final deterministic rendering

**Files:**
- Create: `src/kawaneen/generation/prompt.py`
- Create: `src/kawaneen/generation/rendering.py`
- Test: `tests/test_generation_prompt.py`

**Interfaces:**
- `SYSTEM_PROMPT_VERSION`, `OUTPUT_SCHEMA_VERSION`, `PROMPT_TEMPLATE_VERSION`.
- `render_generation_prompt(query: str, context_pack: ContextPack, jurisdiction_text: str | None = None) -> RenderedPrompt`.
- `generation_version_hash(settings: GenerationSettings) -> str`.
- `render_verified_answer(claims: Sequence[VerifiedClaim], jurisdiction_text: str | None, disclaimer_text: str) -> str`.

- [ ] Write tests proving exact evidence text and IDs are preserved, prompt injection-like evidence is labeled as data, the prompt contains all safety instructions, and the final renderer excludes unverified model metadata.
- [ ] Run the prompt tests RED.
- [ ] Implement stable newline rendering from Phase-9 evidence and SHA-256 versioning over prompt/schema/policy/decoding settings.
- [ ] Run the prompt tests GREEN.

### Task 3: Implement deterministic extractive generation and generator protocol

**Files:**
- Create: `src/kawaneen/generation/generator.py`
- Create: `src/kawaneen/generation/extractive.py`
- Test: `tests/test_generation_extractive.py`

**Interfaces:**
- `Generator` protocol with `generate(request: GenerationRequest) -> GenerationResult`.
- `ExtractiveGenerator.generate(request) -> GenerationResult`.
- `lexical_terms(text: str) -> frozenset[str]`.

- [ ] Write tests for ranking by deterministic overlap, at most two complete source units, exact unmodified quotation text, deterministic tie-breaking, and abstention when no evidence candidate has positive overlap.
- [ ] Run the extractive tests RED.
- [ ] Implement lightweight Arabic/English lexical token extraction without normalization or external model calls; generate claims whose text is exact evidence text and citations point to the matching `E###` evidence.
- [ ] Run the extractive tests GREEN.

### Task 4: Add lazy Ollama and Transformers adapters plus model registry

**Files:**
- Create: `src/kawaneen/generation/ollama.py`
- Create: `src/kawaneen/generation/transformers.py`
- Create: `src/kawaneen/generation/registry.py`
- Test: `tests/test_generation_adapters.py`

**Interfaces:**
- `OllamaGenerator(endpoint: str, model: str, immutable_digest: str | None, transport: OllamaTransport | None = None)`.
- `TransformersGenerator(candidate: ModelCandidate, loader: TransformersLoader | None = None)`.
- `default_model_registry() -> tuple[ModelCandidate, ...]`.
- `lock_ollama_digest(model: str, digest: str) -> ModelCandidate`.

- [ ] Write mocked transport/loader tests proving no imports or network calls occur at construction, locked Ollama digests are required before generation, non-JSON/HTTP errors fail closed without retry, and Transformers loading is lazy.
- [ ] Run adapter tests RED.
- [ ] Implement standard-library localhost HTTP transport and lazy optional imports; never download or resolve weights automatically.
- [ ] Add Qwen primary, Fanar challenger, and generic Jais-compatible candidate metadata with immutable HF revisions represented explicitly.
- [ ] Run adapter tests GREEN.

### Task 5: Implement tokenizer adapters and generation/context budgeting

**Files:**
- Create: `src/kawaneen/generation/tokenizer.py`
- Create: `src/kawaneen/generation/budgeting.py`
- Test: `tests/test_generation_budgeting.py`

**Interfaces:**
- `TokenizerAdapter` protocol, `TokenizerFingerprint`, `CodepointTokenizer`, `LazyHuggingFaceTokenizer`.
- `GenerationBudget(total_input_tokens=3584, output_reservation=384, safety_margin=128)`.
- `budget_context(pack, query, tokenizer, assembler_factory) -> BudgetedContext`.

- [ ] Write tests proving tokenizer fingerprints are stable, prompt overhead is counted before evidence, whole units are omitted, caps are enforced, and retention metrics count only qrels supplied post-hoc.
- [ ] Run budgeting tests RED.
- [ ] Implement lazy tokenizer loading, Qwen tokenizer identity/revision mapping, budget arithmetic, prompt-fit assertion, and text-free per-generator budget reports.
- [ ] Run budgeting tests GREEN.

### Task 6: Add jurisdiction/advice/currentness policy and semantic-support boundary

**Files:**
- Create: `src/kawaneen/generation/policy.py`
- Create: `src/kawaneen/generation/semantic.py`
- Test: `tests/test_generation_policy.py`

**Interfaces:**
- `JurisdictionScope`, `JurisdictionDecision`, `PolicyContext`, `PolicyOutcome`, `evaluate_pre_generation_policy(query, context) -> PolicyOutcome`.
- `SemanticSupport` protocol with `assess(claim_text, evidence_text) -> SemanticAssessment`.
- `DeferredSemanticSupport.assess(...)` returns `available=False`.

- [ ] Write tests for Arabic/English personalized-advice patterns, explicit in-scope/out-of-scope jurisdiction, ambiguous jurisdiction, superseded sources, unavailable currentness status, and deferred semantic support.
- [ ] Run policy tests RED.
- [ ] Implement deterministic refusal/reframe outcomes and stable abstention reasons; never infer jurisdiction from document text or dataset name.
- [ ] Run policy tests GREEN.

### Task 7: Add checkpoints, evaluation metrics, and text-free experiment artifacts

**Files:**
- Create: `src/kawaneen/generation/checkpoints.py`
- Create: `src/kawaneen/generation/evaluation.py`
- Create: `src/kawaneen/generation/artifacts.py`
- Create: `src/kawaneen/generation/experiments.py`
- Test: `tests/test_generation_evaluation.py`

**Interfaces:**
- `CheckpointManifest`, `write_checkpoint`, `load_checkpoint`.
- `evaluate_budget_report`, `evaluate_generation_results`.
- `write_text_free_artifact(path, payload)`, `artifact_fingerprint(payload)`.
- `ExperimentSpec`, `prepare_experiment(spec) -> ExperimentPlan`.

- [ ] Write tests proving checkpoint writes are atomic, artifacts reject source text/quotes, fingerprints are deterministic, evaluations are qrels-post-hoc, and experiment preparation never invokes a generator.
- [ ] Run evaluation tests RED.
- [ ] Implement JSON-safe, text-free, deterministic artifacts and checkpoint metadata with no model execution.
- [ ] Run evaluation tests GREEN.

### Task 8: Focused verification and handoff audit

**Files:**
- Modify: `src/kawaneen/generation/__init__.py` only if exports need final adjustment.
- Test: all `tests/test_generation_*.py` plus existing grounding contract/citation tests.

- [ ] Run `uv run pytest tests/test_generation_contracts.py tests/test_generation_prompt.py tests/test_generation_extractive.py tests/test_generation_adapters.py tests/test_generation_budgeting.py tests/test_generation_policy.py tests/test_generation_evaluation.py tests/test_grounding_contracts.py tests/test_grounding_citations.py -o addopts='' -v`.
- [ ] Run `uv run ruff check` on changed Python files and `uv run ruff format --check` on changed Python files.
- [ ] Run `uv run pyright src/kawaneen/generation src/kawaneen/grounding`.
- [ ] Run text-free artifact and import-side-effect tests; do not run `make check`, full pytest/coverage, Phase-8 retrieval, or holdout.
- [ ] Inspect `git diff` and `git status`; do not commit or push.
