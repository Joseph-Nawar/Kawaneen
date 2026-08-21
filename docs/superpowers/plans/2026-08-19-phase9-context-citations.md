# Phase 9 Context Assembly and Citation Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic, corpus-grounded context assembly and citation/claim verification over the immutable Phase-8 DEV serving top-8 rankings.

**Architecture:** A new `kawaneen.grounding` package owns strict frozen contracts, canonical provenance resolution, assembly/rendering, citation verification, DEV audit metrics, and private/tracked artifacts. The package reads Phase-8 ranking JSON and Phase-6 canonical units/Phase-7 chunk mappings as immutable inputs; it never imports retrieval execution, qrels affect only post-hoc audit metrics, and no model/tokenizer is loaded.

**Tech Stack:** Python 3.11/3.12, Pydantic v2 frozen models, standard-library JSON/hashlib/pathlib, pytest fixtures with deterministic fake token counters, existing argparse CLI and atomic text-free artifact conventions.

**Spec:** User-approved Phase 9 context assembly and citation grounding requirements in the task request.

## Global Constraints

- Consume only persisted frozen Phase-8 DEV reranker rankings, serving depth 8, selection SHA `a62cc772f2b71883355c7935da7e7b87ab4d22b3746553148b4f64ef20f28b0b`.
- Resolve every source field by canonical `chunk_id`; unknown IDs fail closed; never use normalized search text or retrieval metadata as authority.
- Deduplicate by canonical unit ID; preserve exact canonical text; no fuzzy matching, NLI, generation, model loading, Phase-8 rerun, holdout, commit, push, or Phase 10 work.
- Token budgets are enforced only at complete canonical-unit boundaries and include rendered headers, headings, evidence IDs, and text.
- Qrels are post-hoc evaluation inputs only; runtime assembly is label-blind.
- Run only focused tests, Ruff, grounding/retrieval Pyright, deterministic/hash audits, and tracked/private text audits; never `make check` or full pytest/coverage.

### Task 1: Define immutable grounding contracts and failing contract tests

**Files:**
- Create: `src/kawaneen/grounding/contracts.py`
- Create: `src/kawaneen/grounding/__init__.py`
- Test: `tests/test_grounding_contracts.py`

**Interfaces:**
- `SourceRecord`, `CanonicalEvidenceUnit`, `ResolvedChunk`, `RetrievalInput`, `ContextUnit`, `ContextBlock`, `EvidenceReference`, `ContextPack`.
- `CitationRequest`, `VerifiedCitation`, `ClaimDraft`, `GeneratedDraft`, `CitationVerification`, `VerificationResult`.
- All models are frozen Pydantic models with `extra="forbid"`; generator-facing request fields are exactly `evidence_id` and `quoted_text`.

- [ ] Write tests proving models reject extra fields, invalid empty IDs/text, and generator citation metadata.
- [ ] Run `uv run pytest tests/test_grounding_contracts.py -o addopts=''`; verify expected RED failures.
- [ ] Implement the minimal frozen models and enums/protocol-facing types.
- [ ] Re-run the focused contract tests and verify GREEN.

### Task 2: Add canonical provenance resolution and ranking readers

**Files:**
- Create: `src/kawaneen/grounding/provenance.py`
- Create: `src/kawaneen/grounding/inputs.py`
- Test: `tests/test_grounding_provenance.py`

**Interfaces:**
- `CanonicalCorpusResolver.from_json(canonical_path, chunks_path, corpus_manifest_path)`.
- `CanonicalCorpusResolver.resolve_chunk(chunk_id) -> ResolvedChunk`, failing closed for unknown chunks or units.
- `load_frozen_phase8_dev_rankings(...) -> tuple[RetrievalInput, ...]`, validating the immutable selection SHA and reading only `ranked_chunk_ids[:8]` from persisted reranker files.

- [ ] Write tests for authoritative text/metadata, unknown chunk/unit failure, retrieval metadata being ignored, deterministic ranking order, and frozen SHA validation.
- [ ] Run the focused provenance/input tests RED.
- [ ] Implement read-only JSON/JSONL loaders and canonical maps; preserve unit `text` exactly and expose unavailable metadata as `None`.
- [ ] Run the focused tests GREEN without importing retrieval execution or model packages.

### Task 3: Implement deterministic assembly, rendering, and token budgeting

**Files:**
- Create: `src/kawaneen/grounding/assembly.py`
- Create: `src/kawaneen/grounding/rendering.py`
- Test: `tests/test_grounding_assembly.py`

**Interfaces:**
- `TokenCounter` protocol with `identity` and `count(text) -> int`.
- `ContextAssembler(resolver, token_counter, max_context_tokens, assembly_policy_version=...)`.
- `ContextAssembler.assemble(query_id, ranked_inputs, phase8_selection_sha, corpus_hash) -> ContextPack`.
- `render_context(pack) -> str` and `render_evidence(pack) -> str`.

- [ ] Write tests for overlap/exact duplicate chunks, adjacent/non-adjacent units, multiple documents, heading changes, missing metadata, identical text across documents, rank/source-order conflict, deterministic ties, empty retrieval, exact budget, over-budget unit omission, and no duplicate units.
- [ ] Run the assembly tests RED.
- [ ] Implement ascending-rank selection, provenance accumulation, contiguous same-document/same-heading grouping, document-best-rank ordering, canonical ordering, deterministic `E001` IDs, and whole-unit omission records.
- [ ] Implement rendering/counting that accounts for all headers/headings/evidence IDs/text and never truncates.
- [ ] Run assembly tests GREEN and assert zero budget/order/duplicate violations in fixtures.

### Task 4: Implement exact citation and structural claim verification

**Files:**
- Create: `src/kawaneen/grounding/citations.py`
- Create: `src/kawaneen/grounding/verification.py`
- Test: `tests/test_grounding_citations.py`

**Interfaces:**
- `verify_citation(pack, request) -> CitationVerification`.
- `verify_draft(pack, draft) -> VerificationResult`.
- Exact codepoint substring matching against authoritative contributing unit text; match selection is lowest retrieval rank then lexicographically smallest chunk ID.
- Structural-only verification documents semantic entailment as deferred to Phase 10.

- [ ] Write adversarial tests for unknown/out-of-context evidence, invented metadata, different/altered/normalized Arabic quotes, empty quote, unsupported claims, empty context, malformed drafts, and accepted citation metadata.
- [ ] Run the citation tests RED.
- [ ] Implement fail-closed exact matching and server-side citation construction; reject all generator-supplied metadata.
- [ ] Implement claim representation checks, citation validation, abstention rules, and explicit non-entailment status.
- [ ] Run citation tests GREEN and prove accepted citations trace to context and exact authoritative substrings.

### Task 5: Add integrity fingerprints, audit metrics, artifacts, and CLI

**Files:**
- Create: `src/kawaneen/grounding/artifacts.py`
- Create: `src/kawaneen/grounding/evaluation.py`
- Create: `src/kawaneen/grounding/dev.py`
- Modify: `src/kawaneen/cli.py`
- Test: `tests/test_grounding_audit.py`

**Interfaces:**
- `context_pack_fingerprint(pack, phase8_selection_sha, corpus_hash, token_counter, max_context_tokens) -> str`.
- `audit_dev_contexts(...) -> dict[str, object]` with text-free aggregate metrics and GoldEvidenceRetention/CompleteGoldEvidenceRetention.
- `assemble_dev(...)` and `audit_dev(...)` CLI handlers; no model/retrieval calls.
- Private packs under `artifacts/private/phase9_grounding/`; tracked text-free manifests under `data/manifests/grounding/` and `data/evaluation/`.

- [ ] Write audit tests for retention accounting, no-only-representation dedup loss, text-free tracked payloads, stable fingerprints, and CLI parser/dispatch.
- [ ] Run audit tests RED.
- [ ] Implement private JSON pack persistence, text-free tracked writers, deterministic manifests/policy/schema, aggregate counters, and hash fields.
- [ ] Add `grounding assemble-dev` and `grounding audit-dev` without changing retrieval orchestration.
- [ ] Run audit tests GREEN and inspect that no tracked output contains source text or quotes.

### Task 6: Run focused verification and produce the DEV audit

**Files:**
- Create/update: `data/manifests/grounding/phase9_context_policy.json`
- Create/update: `data/manifests/grounding/phase9_citation_schema.json`
- Create/update: `data/evaluation/phase9_dev_context_audit.json`
- Create/update: `data/evaluation/phase9_citation_integrity_audit.json`
- Create/update: `data/evaluation/phase9_grounding_report.json`
- Private: `artifacts/private/phase9_grounding/`

- [ ] Run focused grounding/retrieval tests with coverage disabled via `-o addopts=''`.
- [ ] Run Ruff on changed Python files.
- [ ] Run Pyright on grounding and retrieval source paths only.
- [ ] Run `kawaneen grounding assemble-dev` and `audit-dev` against persisted Phase-8 DEV artifacts; do not pass qrels into assembly.
- [ ] Run deterministic/hash and tracked/private text audits; record command outputs, hashes, unavailable metadata, and blockers.
- [ ] Confirm no full repository pytest/coverage, `make check`, model execution, holdout, commit, or push.
