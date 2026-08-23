# Phase 10 Stage A — Local Generation and Abstention Infrastructure

## Scope and invariants

Phase 10 Stage A adds local generation infrastructure on top of the immutable
Phase-9 `ContextPack` and citation verifier. It stops before real generative,
Transformers, or NLI inference: adapters are implemented lazily and tested
with injected fakes. Phase-8 retrieval, Phase-9 grounding guarantees, qrels,
holdout artifacts, and upstream source text are not modified.

The LLM is an untrusted claim proposer. The only accepted model payload is a
strict JSON object containing a decision and zero to three claims. Each claim
contains only claim text and context-local evidence IDs with exact quotations.
Document metadata, jurisdiction, URLs, titles, articles, pages, answers, and
reasoning are rejected at the model boundary. Phase 9 remains responsible for
resolving citation metadata and checking exact source substrings.

## Components

`kawaneen.generation.contracts` owns frozen Pydantic contracts for model
outputs, generation settings, registry entries, jurisdiction decisions,
policy outcomes, abstention reasons, and generation results.

`prompt` and `rendering` version the system prompt and deterministic prompt
serialization. They render the query and Phase-9 evidence without rewriting
evidence text, and hash the prompt/schema/policy/decoding versions.

`generator` defines the synchronous generator protocol and common failure
behavior. `extractive` provides a deterministic no-model baseline that selects
at most two complete canonical evidence units using query/evidence lexical
overlap and returns exact source text only. `ollama` uses localhost HTTP only,
requires an immutable digest for execution, and has no retry. `transformers`
imports Transformers lazily and accepts injected model/tokenizer factories so
normal tests do not install weights or access the network.

`tokenizer` defines tokenizer adapters and fingerprints. The default registry
locks the matching Qwen tokenizer identity and revision without downloading
weights. `budgeting` computes prompt overhead first, reserves 384 output tokens
and 128 safety tokens, and delegates evidence selection to a Phase-9
assembler configured with the remaining input budget. It records prompt and
evidence counts, omitted units, and budgeted GoldEvidenceRetention metrics.

`policy` and `abstention` apply deterministic jurisdiction, personalized
advice, superseded-source, currentness, confidence, conflict, and missing
information gates before generation and after invalid output. `semantic`
defines a deferred semantic-support interface; it cannot silently claim NLI
support in Stage A.

`checkpoints`, `evaluation`, and `artifacts` provide text-free, deterministic
state and report structures for later experiments. No Stage-A command runs
Phase-10 holdout or model inference.

## Data flow

1. A caller supplies a query, an immutable Phase-9 `ContextPack`, and server
   policy inputs.
2. Pre-generation policy validates jurisdiction and safety conditions.
3. Token budgeting renders the versioned prompt and, when assembling from
   ranked evidence, allocates whole-unit evidence budget through the Phase-9
   assembler.
4. A generator proposes strict JSON. Malformed JSON or schema violations fail
   closed to `INVALID_GENERATION`.
5. The application validates every claim/citation structurally with the Phase-9
   verifier and consults the deferred semantic-support interface.
6. Only verified claims are available to the deterministic final renderer;
   server-controlled jurisdiction and disclaimer text are added outside the
   model.

## Registry and safety policy

The primary candidate is `Qwen/Qwen3-4B-Instruct-2507` at an explicitly locked
Hugging Face revision, with matching Ollama name
`qwen3:4b-instruct-2507-q4_K_M`. `QCRI/Fanar-1-9B-Instruct` is an optional
challenger. Jais is supported by the generic Transformers adapter but is not
an initial experiment. No weights are downloaded by registry construction.

The existing source/corpus records do not provide an authoritative active
single-jurisdiction field for retrieved documents. Stage A therefore defaults
to `unverified`; it never infers jurisdiction from document text or dataset
names. Explicit out-of-scope jurisdiction requests fail before generation.
Personalized Arabic and English legal-advice requests are refused or
deterministically reframed as general information. Explicit currentness
questions abstain when source status is unavailable.

## Testing and verification

Tests are focused under `tests/test_generation_*.py` and use no model weights,
network, or real Ollama server. Verification is limited to focused generation
and grounding tests, Ruff on changed files, and Pyright on generation and
grounding packages. The full repository suite, coverage gate, `make check`,
Phase-8 retrieval, and holdout execution are explicitly out of scope.
