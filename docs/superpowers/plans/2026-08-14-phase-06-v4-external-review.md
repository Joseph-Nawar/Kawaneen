# Phase 6 v4 External Review Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the authoritative v3 external source-review adjudication as a bounded record-level transformation and produce the private `phase6-retrieval-eval-draft-v4` review candidate without freezing it.

**Architecture:** Read v3 items and the 240-record adjudication, preserve accepted unanswerables and correctable evidence spans, replace only explicit `replace` bases, then regenerate all variants from the corrected 200-base set. Reuse the existing canonical corpus, semantic-target models, chunk mapper, split/leakage planner, review packet, diagnostics, and handoff utilities; write only a new ignored `draft-v4` artifact root plus text-free summaries.

**Tech Stack:** Python 3.12, Pydantic models, JSONL, pytest/coverage, Ruff, Pyright, pre-commit, Make.

## Global Constraints

- Do not modify the frozen corpus scope/hash, canonical corpus, Phase-5 `legal-structure-v1` policy, governance, privacy architecture, schema, holdout guard, or earlier-phase artifacts.
- Do not use retrieval scores or rankings.
- Do not freeze, commit, push, or begin Phase 7.
- Keep all text-bearing v4 artifacts under `artifacts/private/phase6_evaluation/draft-v4/`.
- Keep every record `human_verified=false`; the external AI review is not human verification.
- Preserve exactly 25 accepted base unanswerables; apply exactly 118 `correct`, 57 `replace`, and 40 `regenerate_variant` decisions.

---

### Task 1: Add failing adjudication and semantic-regression tests

**Files:**
- Create: `tests/test_evaluation_v4_adjudication.py`
- Modify: `tests/test_evaluation_v3_semantic_targets.py`

**Interfaces:**
- Consumes: v3 private selected items and `/Users/nawar/Downloads/phase6_v3_external_ai_review_adjudication.jsonl`.
- Produces: executable assertions for disposition counts, unchanged accepted evidence, correct-vs-replace scope, semantic target rules, all-record duplicate checks, and variant language gates.

- [ ] **Step 1: Write tests** for exact decision counts; accepted unanswerable preservation; corrected-span identity; replacement-only scope; deterministic old→new mapping; holding queries without disposition; multi queries without conclusion; deadline date rejection; heading/fragment definition rejection; authority actor/power matching; condition negation preservation; partial disposition preservation; no blank English slots; and duplicate detection across all 240 records.
- [ ] **Step 2: Run the focused tests** with `uv run pytest -q tests/test_evaluation_v4_adjudication.py tests/test_evaluation_v3_semantic_targets.py -o addopts=''` and confirm they fail for the missing v4 implementation.

### Task 2: Implement bounded v3→v4 record transformation

**Files:**
- Create: `src/kawaneen/evaluation/adjudication_v4.py`
- Modify: `src/kawaneen/evaluation/models.py` only if a typed correction/mapping model is required by existing conventions.
- Modify: `src/kawaneen/evaluation/semantic_targets.py`
- Modify: `src/kawaneen/evaluation/candidates_v3.py` only for reusable corrected semantic extractors or ArabiCCR multi-evidence construction.

**Interfaces:**
- `load_v3_adjudication(path: Path) -> tuple[ExternalReviewDecision, ...]`
- `apply_v3_adjudication(v3_items: tuple[DatasetItem, ...], corpus: EvaluationCorpus, decisions: tuple[ExternalReviewDecision, ...], pool: tuple[DatasetItem, ...]) -> V4BuildResult`
- `validate_adjudication_application(...) -> ...`
- `regenerate_v4_variants(base_items: tuple[DatasetItem, ...]) -> tuple[DatasetItem, ...]`

- [ ] **Step 1: Implement decision parsing** with exact count validation and rejection of missing, duplicate, unknown, or reinterpreted dispositions.
- [ ] **Step 2: Implement accepted/correct handling** preserving accepted unanswerable records and exact evidence spans for every correct record, while deriving corrected typed targets, natural queries, concise answers, and deterministic IDs.
- [ ] **Step 3: Implement category-specific fail-closed corrections** for identifiers/effects, definitions, deadlines/triggers, actor-power relations, condition polarity, neutral holding issues, and genuinely necessary multi-evidence premises.
- [ ] **Step 4: Implement replacement selection** from the existing evidence-qualified pool only for the 57 explicit base replacements, with category quotas, source/opportunity ordering, no retrieval input, and concrete rejection reasons for failed ArabiCCR multi-evidence candidates.
- [ ] **Step 5: Implement deterministic IDs, evidence-preserving/replacing mapping records, chunk remapping, and variant regeneration only after all bases validate.
- [ ] **Step 6: Run focused tests and confirm green.

### Task 3: Add v4 orchestration and private artifacts

**Files:**
- Modify: `src/kawaneen/evaluation/orchestrator.py`
- Modify: `src/kawaneen/cli.py`
- Modify: `Makefile`
- Modify: `docs/phases/phase-06-retrieval-evaluation-dataset.md`

**Interfaces:**
- `run_build_draft_v4(review_file: Path, ...) -> dict[str, object]`
- CLI: `kawaneen evaluation build-draft-v4 --review-file <path>`.

- [ ] **Step 1: Add the v4 command** without changing v3 artifacts or freeze behavior.
- [ ] **Step 2: Emit private `draft-v4/draft/selected_and_variants.jsonl`, review packet, diagnostics, adjudication application mapping, and source context handoff.
- [ ] **Step 3: Emit text-free v4 summary with corpus/hash/policy IDs, counts, distributions, decision application counts, validation summaries, replacement reasons, and private paths.
- [ ] **Step 4: Ensure all v4 records remain draft and `human_verified=false`.

### Task 4: Deterministic verification and handoff

**Files:**
- Modify tests/docs only as needed to encode verified behavior.

- [ ] **Step 1: Run v4 validation, privacy, span/qrel, duplicate/near-duplicate, split leakage, corpus hash, and tracked/private audits.
- [ ] **Step 2: Run the v4 build twice and compare item, split, evidence/qrel, review-state, and policy hashes.
- [ ] **Step 3: Run `pytest` with coverage, Ruff, Pyright, pre-commit, and `make check`.
- [ ] **Step 4: Confirm freeze remains blocked on human-review gates; do not freeze, commit, or push.
