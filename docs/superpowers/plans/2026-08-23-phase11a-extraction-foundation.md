# Phase 11A Structured Regulatory Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the private, source-grounded Phase 11A regulatory extraction foundation with deterministic extraction, validated semantic proposals, annotation readiness, protected evaluation, and text-free readiness artifacts.

**Architecture:** The new `kawaneen.extraction` package owns strict contracts and orchestration while reusing canonical corpus models, source registry loading, local-only tokenizer loading, and text-free artifact hashing. Deterministic candidates are request-local and exact-span based; the hybrid assembler accepts only validated Qwen proposals and never performs model inference in this phase.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, PyArrow, argparse, pytest, Ruff, Pyright, existing `uv` commands.

**Spec:** Approved Phase 11A request in the conversation.

## Global Constraints

- `saudi-moj-derived` is the Phase-11 v1 regulatory extraction universe; case-law sources are ineligible through governed metadata and canonical unit type.
- Preserve 120 total / 80 DEV / 40 protected HOLDOUT / 10 DEV-smoke, with document-disjoint DEV/HOLDOUT splits.
- Weak deterministic cues are sampling strata only, never gold labels.
- Raw statutory text and annotations are private; tracked artifacts are text-free.
- Issuing authority comes only from governed metadata and is null when unavailable.
- Qwen proposals contain exact source spans and deterministic candidate IDs only; unsupported or ambiguous spans fail closed.
- Exact canonical codepoint offsets and original Arabic text are preserved; normalization is additive.
- Field-level provenance must identify metadata, deterministic, or llm_selected origin.
- Checkpoints use explicit lifecycle semantics and cannot treat readiness as completed hybrid extraction.
- HOLDOUT remains sealed during Phase 11A; ordinary DEV commands cannot access it.
- No Qwen/Ollama calls, model downloads, training, HOLDOUT evaluation, full pytest, coverage, polling, commit, or push.

---

### Task 1: Establish strict extraction contracts and exact span security

**Files:**
- Create: `src/kawaneen/extraction/contracts.py`
- Create: `src/kawaneen/extraction/span_validation.py`
- Create: `src/kawaneen/extraction/__init__.py`
- Test: `tests/test_extraction_contracts.py`
- Test: `tests/test_extraction_spans.py`

**Interfaces:**
- Produces `ExactSourceSpan`, `Candidate`, `CandidateRegistry`, `NormativeRule`, `SemanticProposal`, `ExtractionResult`, `FieldProvenance`, and strict enums.
- Produces `resolve_exact_span(text, proposed_text, occurrence=None)` and `validate_candidate_reference(registry, candidate_id)`.

- [ ] Write tests for exact reconstruction, ambiguity rejection, forbidden fields, proposal restrictions, and final schema fields.
- [ ] Run `uv run pytest tests/test_extraction_contracts.py tests/test_extraction_spans.py -q`; confirm the new imports fail before implementation.
- [ ] Implement frozen Pydantic v2 models with `extra="forbid"`, exact offset validation, candidate ID patterns, provenance, validation diagnostics, and final extraction field groups.
- [ ] Run the focused tests again and confirm they pass.

### Task 2: Implement additive Arabic normalization and deterministic candidates

**Files:**
- Create: `src/kawaneen/extraction/normalization.py`
- Create: `src/kawaneen/extraction/candidates.py`
- Create: `src/kawaneen/extraction/deterministic.py`
- Test: `tests/test_extraction_candidates.py`

**Interfaces:**
- Produces `normalize_numeric`, `normalize_date`, `normalize_duration`, and `build_candidate_registry`.
- Candidate IDs are assigned by source order as `T001`, `M001`, `P001`, `A001`, and `R001`, deduplicating identical spans only.

- [ ] Write tests for ASCII/Arabic-Indic/Persian digits, separators, SAR/ريال, percentages, Gregorian/Hijri dates, durations, article/regulation references, ordering, duplicate suppression, and canonical text preservation.
- [ ] Run the focused candidate test and confirm it fails for missing implementations.
- [ ] Implement conservative regex extraction with explicit partial/unresolved normalization statuses and preserved calendar/components/raw text.
- [ ] Run the focused candidate test and confirm it passes.

### Task 3: Add governed source policy, private annotation selection, and annotation validation

**Files:**
- Create: `src/kawaneen/extraction/source_policy.py`
- Create: `src/kawaneen/extraction/annotation.py`
- Create: `src/kawaneen/extraction/artifacts.py`
- Test: `tests/test_extraction_governance.py`
- Test: `tests/test_extraction_dataset.py`

**Interfaces:**
- Produces `eligible_regulatory_unit`, `select_annotation_units`, `validate_annotation_record`, and `prepare_annotation_pack`.
- Reads canonical Parquet and the governed source registry; writes private records under `artifacts/private/phase11_extraction/annotations/` and text-free manifests under `data/manifests/extraction/`.

- [ ] Write tests for statutory eligibility, case-law exclusion, authority null behavior, exact 120/80/40/10 counts, document disjointness, reproducibility, sealed HOLDOUT, unreviewed state, and no text in tracked manifests.
- [ ] Run focused governance/dataset tests and confirm the expected failures.
- [ ] Implement deterministic weak-cue strata, corpus-hash seeded selection, document-level split assignment, private annotation records, protected holdout access gates, and text-free manifest writing.
- [ ] Run focused governance/dataset tests and confirm they pass.

### Task 4: Implement semantic proposal schema, prompt, provider boundary, and hybrid assembly

**Files:**
- Create: `src/kawaneen/extraction/prompt.py`
- Create: `src/kawaneen/extraction/provider.py`
- Create: `src/kawaneen/extraction/hybrid.py`
- Test: `tests/test_extraction_hybrid.py`

**Interfaces:**
- Produces `semantic_proposal_schema`, `render_extraction_prompt`, `ExtractionProvider`, `MockExtractionProvider`, and `assemble_hybrid_result`.
- The provider boundary accepts raw JSON but never loads a model; the hybrid assembler validates JSON, Pydantic structure, candidate IDs, and exact canonical spans before acceptance.

- [ ] Write tests for obligations, prohibitions, permissions, exceptions, penalties, multiple rules, empty result, malformed JSON, metadata rejection, unsupported spans, ambiguity, invalid candidate IDs, and mocked provider calls.
- [ ] Run the focused hybrid tests and confirm they fail before implementation.
- [ ] Implement the compact strict proposal schema, pinned model/tokenizer metadata, local-only tokenizer preflight adapter, fail-closed field dropping with diagnostics, and provenance-complete final assembly.
- [ ] Run the focused hybrid tests and confirm they pass with zero real provider calls.

### Task 5: Add protected checkpoint lifecycle, deterministic evaluation, and readiness reporting

**Files:**
- Create: `src/kawaneen/extraction/checkpoints.py`
- Create: `src/kawaneen/extraction/evaluation.py`
- Create: `src/kawaneen/extraction/readiness.py`
- Test: `tests/test_extraction_evaluation.py`
- Test: `tests/test_extraction_runtime.py`

**Interfaces:**
- Produces `ExtractionCheckpointStore`, `evaluate_extractions`, `classify_errors`, and `build_readiness_report`.
- Checkpoint fingerprints include source/unit hash, extractor configuration, candidate version, prompt hash, schema hash, model/digest, tokenizer revision, and semantic policy.

- [ ] Write tests for TP/FP/FN, zero support, boundary/modality errors, normalized money equality, micro/macro metrics, clause exact match, rule metrics, error categories, lifecycle resume, incomplete recomputation, fingerprint rejection, local-only tokenizer behavior, and status without model loading.
- [ ] Run the focused runtime/evaluation tests and confirm failure on missing implementations.
- [ ] Implement strict span metrics, rule metrics, engineering/safety rates, aggregate error categories, lifecycle validation, and text-free readiness aggregation.
- [ ] Run the focused runtime/evaluation tests and confirm they pass.

### Task 6: Add CLI commands and documentation

**Files:**
- Modify: `src/kawaneen/cli.py`
- Modify: `Makefile`
- Create: `docs/phase11-extraction.md`
- Test: `tests/test_extraction_cli.py`

**Interfaces:**
- Adds `kawaneen extraction status`, `prepare-annotations`, `validate-annotations --split`, `run-deterministic --split`, `run-hybrid --split [--resume]`, and protected `evaluate --extractor --split [--allow-holdout]`.
- Status is metadata-only and does not load Qwen or the entire corpus.

- [ ] Write parser and command tests for DEV behavior, explicit HOLDOUT protection, and hybrid non-execution.
- [ ] Run the focused CLI tests and confirm they fail before command wiring.
- [ ] Wire the commands to extraction orchestration with clear refusal errors for HOLDOUT and hybrid execution in this task.
- [ ] Document architecture, schema, grounding, annotations, metrics, holdout protocol, privacy, and limitations.
- [ ] Run the focused CLI tests and confirm they pass.

### Task 7: Generate and audit Phase 11A readiness artifacts

**Files:**
- Create: `data/manifests/extraction/phase11_annotation_selection.json`
- Create: `data/manifests/extraction/phase11_readiness.json`
- Create: `data/evaluation/phase11_readiness.json`
- Create: `artifacts/private/phase11_extraction/annotations/`
- Test: `tests/test_extraction_readiness.py`

- [ ] Run the deterministic selection/readiness command once against the existing local statutory corpus.
- [ ] Run focused extraction tests once, Ruff on changed files, targeted Pyright, and text/privacy/hash audits.
- [ ] Verify the report contains exact selection counts, strata and candidate distributions, governance state, annotation state, zero Qwen calls, and readiness-manifest SHA-256.
- [ ] Confirm no hybrid inference, HOLDOUT evaluation, full pytest, coverage, commit, or push occurred.

