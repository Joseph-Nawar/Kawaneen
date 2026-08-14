# Phase 6 AI-Reviewed Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reclassify the existing literal-patch final candidate as an externally AI-reviewed, engineering-ready Phase 6 release without changing private record text.

**Architecture:** Add a release-freeze path that copies the existing final-candidate private artifacts into a versioned AI-reviewed release, verifies byte identity and all existing hashes, and emits only text-free tracked metadata. Update methodology and freeze documentation to distinguish AI review from human verification while preserving human review as an optional upgrade.

**Tech Stack:** Python 3.12, Pydantic, Typer, pytest, uv, Ruff, Pyright, pre-commit, Make.

## Global Constraints

- Preserve the 240 final records exactly as produced by the literal patch.
- Keep `human_verified=false` and `review_provenance=independent_ai_source_review`.
- Do not generate, correct, review, adjudicate, or rewrite private evaluation text.
- Preserve corpus, item, split, evidence/qrel, policy, and review hashes.
- Do not describe the release as human-gold, expert-reviewed, or human-annotated.
- The release must not be frozen by the existing human-review gate; future publication-grade human review remains optional.
- Do not commit private artifacts; only text-free tracked metadata, code, tests, and docs may be committed.

### Task 1: Add failing release-freeze tests

**Files:**
- Test: `tests/test_evaluation_ai_reviewed_freeze.py`

- [ ] Add tests proving the release command copies private files byte-for-byte, emits the requested version and text-free manifest/report, preserves review provenance and `human_verified=false`, and does not require human-review completion.
- [ ] Run the focused tests and confirm they fail because the release command is not implemented.

### Task 2: Implement the AI-reviewed release freeze

**Files:**
- Modify: `src/kawaneen/evaluation/orchestrator.py`
- Modify: `src/kawaneen/cli.py`
- Modify: `Makefile`

- [ ] Implement a bounded release function that reads only the existing final candidate, verifies its record count, exact corpus hash, review metadata, and existing summary hashes, then copies private artifacts without changing bytes.
- [ ] Add a CLI command and Make target for creating `phase6-retrieval-eval-ai-reviewed-v1`.
- [ ] Make the generated release manifest/report text-free while including counts, distributions, hashes, review provenance, limitation, and validation results.
- [ ] Run focused tests and confirm they pass.

### Task 3: Update methodology and freeze policy

**Files:**
- Modify: `docs/phases/phase-06-retrieval-evaluation-dataset.md`
- Modify: `data/evaluation/README.md`

- [ ] Document the externally AI-reviewed classification and explicit limitation.
- [ ] State that human review remains an optional future publication-grade upgrade and does not block the current engineering Phase 7 gate.
- [ ] Add the release version and command without describing it as human-gold or human-verified.

### Task 4: Run full verification and publish the branch

**Files:**
- Modify: `data/manifests/evaluation/phase6_ai_reviewed_v1_manifest.json`
- Modify: `data/evaluation/phase6_retrieval_eval_ai_reviewed_v1_report.json`

- [ ] Run full pytest/coverage, Ruff, Pyright, pre-commit, `make check`, deterministic/hash checks, and Git/private-artifact audits.
- [ ] Verify the private release dataset remains byte-identical to final-candidate-v1.
- [ ] Commit only code/docs/text-free metadata and push the current branch.
