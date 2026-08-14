# Phase 6 Retrieval Evaluation Dataset Implementation Plan

> For agentic workers: this plan is executed inline in the current Phase 6 branch. No commit or push is permitted.

Goal: Build a full governed ALARB + ArabiCCR, text-free-scoped retrieval-evaluation corpus manifest and a private, deterministic, review-gated 240-record draft dataset with canonical-span gold evidence and Phase-5 chunk mappings.

Architecture: Add a typed kawaneen.evaluation package. A corpus module loads the complete Phase-3 canonical Parquet sources, applies the Phase-5 content policy, snapshots source/canonical hashes before any candidate generation, and writes a private text-bearing corpus snapshot plus tracked aggregate metadata. A candidate module creates deterministic document-derived base candidates from evidence-first source spans, selects 200 base intents using structural quality checks only, and adds 40 language/register variants sharing their base intent/evidence. Validation, graph-based split planning, review import/export, and freeze gating live in focused modules; all text-bearing outputs remain below artifacts/private/phase6_evaluation/.

Tech Stack: Python 3.12 (>=3.11,<3.13), Pydantic/dataclasses, existing canonical Parquet readers, SourceSpan/CitationAnchor/LegalChunk, JSONL/CSV private artifacts, argparse CLI, pytest/Ruff/Pyright.

## Global Constraints

- Preserve Phase 3 canonical IDs, hashes, provenance, and source governance.
- Use full eligible ALARB + ArabiCCR scope; never define the final scope from the Phase-5 3,000-document experiment subset.
- Include ALARB facts/court reasoning/applicable laws/verdict and ArabiCCR EVENTS/REASONING/RULING, with case_text only when structured content is unavailable.
- Exclude OCR-derived material and MOJ-derived statute seed from retrieval gold.
- Do not use retrieval performance to construct, select, reject, relabel, rewrite, or split candidates.
- Gold evidence is canonical source spans; chunk qrels are deterministic derivatives.
- Keep all queries, answers, spans, qrels, excerpts, review packets, and frozen records private and ignored.
- Require explicit allow_holdout=True to expose holdout records.
- Never infer or auto-set human verification; freeze remains blocked until review gates pass.
- Do not commit or push.

## Task 1: Establish typed evaluation schema

Files: create src/kawaneen/evaluation/__init__.py, src/kawaneen/evaluation/models.py, and tests/test_evaluation_models.py.

Implement enums/models for query categories, languages/registers, creation methods, answerability/unanswerable reasons, difficulty, review states, split, relevance grades, source spans/evidence groups, review metadata, dataset items, corpus scope, and validation summaries. Enforce deterministic IDs, grades 0/1/2, span bounds, required evidence/gold answer for answerable records, zero evidence/chunks for unanswerable records, variant/base intent consistency, and frozen model immutability.

## Task 2: Freeze the full retrieval corpus manifest

Files: create src/kawaneen/evaluation/corpus.py and tests/test_evaluation_corpus.py.

Load full canonical ALARB/ArabiCCR units and apply source-specific content policy, including structured ArabiCCR precedence and case_text fallback. Reject MOJ/OCR content. Hash canonical files against the Phase-3 inventory, capture source revisions and document/unit inventories, compute a content-policy hash, and write a private text-bearing corpus snapshot before candidate generation. Emit only counts/hashes/distributions to data/manifests/evaluation/phase6_corpus_summary.json.

## Task 3: Build evidence-first private candidates

Files: create src/kawaneen/evaluation/candidates.py, src/kawaneen/evaluation/serialization.py, and tests/test_evaluation_candidates.py.

Generate 260 deterministic base candidates from canonical evidence spans using source-relative templates and structural/manual difficulty. Allocate 39/33/26/26/39/33/32/32 candidates across the eight requested categories, then retain 200 base intents using only non-retrieval structural checks and add 40 variants (10 simple Arabic, 10 Egyptian Arabic, 10 English, 10 Arabic-English code-switched). Store private JSONL item records, evidence groups, qrels, and source excerpts. Record benchmark provenance as unavailable unless an actual permitted benchmark query/relevance file exists; use document_derived only.

## Task 4: Validate privacy, evidence, IDs, and chunk mappings

Files: create src/kawaneen/evaluation/validation.py, src/kawaneen/evaluation/chunks.py, and tests/test_evaluation_validation.py.

Implement exact span checks against the private canonical snapshot, evidence-group and answerability rules, deterministic ID/duplicate/near-duplicate diagnostics, query/source lexical overlap diagnostics, privacy scanning, corpus/canonical hash checks, authoritative-article support checks, and deterministic mapping from evidence spans to current legal-structure-v1 chunks. Keep all source-bearing failures private and write only sanitized validation summaries.

## Task 5: Implement review packets and state transitions

Files: create src/kawaneen/evaluation/review.py and tests/test_evaluation_review.py.

Export JSONL/CSV review packets with query, answer, source excerpts, highlighted evidence, citations, and editable review fields. Import explicit reviewer decisions while preserving human_verified=false unless a reviewer explicitly supplies a verified attestation; reject invalid state transitions and unresolved conflicts. Provide status and agreement summaries for answerability, span overlap, and citation anchors. Freeze gates require primary review for all, independent holdout review for all, independent dev recheck for at least 25%, double review for all unanswerable/hard/multi-evidence items, and zero unresolved disagreements.

## Task 6: Build leakage-safe provisional splits and gated freeze

Files: create src/kawaneen/evaluation/splits.py, src/kawaneen/evaluation/orchestrator.py, tests/test_evaluation_splits.py, and tests/test_evaluation_orchestrator.py.

Group items by base intent, source documents, and multi-document connected components; deterministically assign approximately 160/80 dev/holdout, keeping variants and connected documents together. Mark 20 dev smoke IDs. Implement plan, build-draft, export-review, import-review, validate, freeze, and stats. The loader defaults to dev and rejects holdout access without allow_holdout=True; freeze writes an immutable private v1 bundle only after all human gates pass and emits text-free hashes/counts.

## Task 7: CLI, docs, and verification

Files: modify src/kawaneen/cli.py, Makefile, and data/evaluation/README.md; create docs/phases/phase-06-retrieval-evaluation-dataset.md and tests/test_evaluation_cli.py.

Wire CLI commands and Make targets, document statutory limitations, benchmark availability, schema, category/variant targets, evidence/relevance, review protocol, split leakage controls, privacy, frozen-version policy, and Phase-7 holdout protocol. Add private-path and tracked-artifact audits. Run focused tests, full pytest/coverage, Ruff, Pyright, pre-commit, make check, deterministic rebuild/split/validation checks, canonical hash checks, and Git audits.
