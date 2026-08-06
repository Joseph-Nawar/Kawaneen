# Phase 1 — Legal Data-Source and Licensing Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, offline, fail-closed source-governance layer and a documented legal data-source audit without downloading or ingesting source data.

**Architecture:** `kawaneen.sources` owns typed registry records, CSV loading, cross-record policy validation, and summary generation. Source-specific facts live only in the version-controlled CSV and one evidence note per source; runtime code has no URLs, scraping, networking, or source-name branches. The CLI and Makefile expose validation and summary operations.

**Tech Stack:** Existing Python 3.12/`>=3.11,<3.13`, uv, Pydantic, Python standard library `csv`/`argparse`/`json`/`pathlib`, pytest, Ruff, Pyright, and pre-commit.

## Global Constraints

- Do not commit full datasets, case text, court records, tokens, correspondence, or generated build artifacts.
- Do not implement ingestion, parsing, OCR, NLP, retrieval, APIs, or models.
- Do not use network access in runtime code or tests.
- Use four-state permissions: `yes`, `no`, `conditional`, `unknown`.
- Missing or conflicting evidence must fail closed to `blocked_pending_review`, `metadata_only`, or `excluded`.
- Distinguish paper, code, dataset, and original-source rights; authority does not imply reuse permission.
- Keep the existing runtime dependency set unchanged.

---

### Task 1: Governance model and tests

**Files:** Create `src/kawaneen/sources/__init__.py`, `src/kawaneen/sources/models.py`, `src/kawaneen/sources/registry.py`, `tests/test_sources.py`, and `tests/test_source_cli.py`.

- [ ] Write failing tests for enum/model construction, CSV loading, duplicate IDs, missing evidence, positive-permission evidence, privacy conflicts, role conflicts, conditional conditions, blocked manual actions, missing registry files, summary counts, and CLI text/JSON output.
- [ ] Run the focused tests and confirm they fail because the source package and CLI commands do not exist.
- [ ] Implement minimal typed enums, Pydantic `SourceRecord`, CSV loading, cross-record validation, and summaries with no network or source-specific runtime branches.
- [ ] Run focused and complete tests; preserve branch-aware coverage above 85%.

### Task 2: CLI and Makefile integration

**Files:** Modify `src/kawaneen/cli.py` and `Makefile`; modify `tests/test_source_cli.py` as needed.

- [ ] Add `kawaneen sources validate`, `kawaneen sources summary`, and `kawaneen sources summary --format json`.
- [ ] Add `make sources-validate` and `make sources-summary`; include registry validation in `make check`.
- [ ] Ensure missing files and invalid registries return non-zero results without creating files or accessing the network.

### Task 3: Registry and evidence

**Files:** Create `data/manifests/source_registry.csv`, `data/manifests/README.md`, and one Markdown record under `docs/source-audits/` for each registry source.

- [ ] Record the requested 12 source candidates with explicit evidence URLs, verification date, provenance, licence evidence, terms, access, privacy, role, decision, and manual action fields.
- [ ] Use conservative decisions: no dataset redistribution approval without explicit dataset rights, no automated access approval where prohibited, and no public-demo-safe classification for high-PII records without mitigation.
- [ ] Keep all records metadata-only and offline; do not download or embed source contents.

### Task 4: Governance documentation

**Files:** Create `docs/data-governance.md`, `docs/data_sources.md`, `docs/adr/0004-source-governance-and-fail-closed-licensing.md`, `docs/phases/phase-01-data-source-audit.md`, and `docs/reports/phase-01-data-source-audit-report.md`.

- [ ] Document the evidence hierarchy, rights distinctions, decision policy, privacy findings, primary-corpus/evaluation proposal, unresolved manual actions, and explicit non-goals.
- [ ] Document final registry decision counts, files, tests, coverage, commands, outputs, and justified deviations.

### Task 5: Verification and staging

- [ ] Run every requested uv, CLI, quality, Make, pre-commit, and diff check command.
- [ ] Run `git diff --cached --check` after staging all finished files.
- [ ] Verify no credentials, paths, caches, environments, datasets, or generated artifacts are staged; do not commit or push.
