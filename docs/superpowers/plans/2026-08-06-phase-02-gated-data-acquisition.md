# Phase 2 — Gated Data Acquisition, Integrity Validation, and Privacy Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal at authoring time:** Add a fail-closed, source-specific acquisition and inspection layer for the then-approved ALARB and ArabiCCR purposes, while preserving immutable raw bytes and producing deterministic integrity, duplicate, privacy, and manifest evidence. The current qualified snapshot also includes the separately gated MOJ-derived seed.

**Architecture:** `kawaneen.acquisition` is split into typed specifications/policies, secure storage, adapters, integrity/privacy analysis, manifests, and orchestration. The CLI calls orchestration functions; adapters never decide authorization. Offline fixtures exercise all filesystem and validation paths, while live Hub acquisition is opt-in and never runs in ordinary checks.

**Tech Stack:** Python 3.12-compatible standard library, Pydantic, `huggingface_hub`, `pyarrow`, argparse, TOML specifications, CSV/JSON manifests, and pytest fixtures. No Pandas, DVC, Git LFS, OCR, parsing, normalization, retrieval, models, or vector databases.

## Global Constraints

- Supported Python remains `>=3.11,<3.13`; local interpreter is Python 3.12.
- Historical initial-plan constraint: only ALARB and ArabiCCR were in scope for acquisition at plan authoring time. The current qualified snapshot additionally includes the explicitly gated `saudi-moj-derived` seed; all other registry sources remain denied.
- ALARB is limited to local evaluation, integrity, duplicate, and privacy inspection.
- ArabiCCR is limited to local research and inspection; no train/test split is created.
- Training, publishing, public display, and public-demo operations remain denied.
- Raw files are ignored, source/version namespaced, immutable by convention, and never modified in place.
- No blocked, metadata-only, conditional-unresolved, or excluded source may be downloaded.
- All tests are deterministic and offline unless explicitly marked `network`.
- No raw legal text, identifiers, credentials, private review bundle, cache, or build output may be staged.

## File map

- Create `src/kawaneen/acquisition/models.py` for purposes, operations, specifications, file expectations, hashes, integrity/privacy findings, and manifests.
- Create `src/kawaneen/acquisition/specs.py` for TOML loading and strict specification validation.
- Create `src/kawaneen/acquisition/policy.py` for registry-driven authorization with no bypass argument.
- Create `src/kawaneen/acquisition/storage.py` for repository-relative path safety, symlink checks, `.partial` cleanup, hashing copies, and atomic moves.
- Create `src/kawaneen/acquisition/integrity.py` for byte hashes, CSV/Parquet checks, schema fingerprints, and duplicate analysis.
- Create `src/kawaneen/acquisition/privacy.py` for deterministic masked findings and private-review bundle generation under ignored artifacts.
- Create `src/kawaneen/acquisition/adapters.py` for the Hub adapter and safe local-file adapter; expose an ArabiCCR public API adapter only if its documented public route is verified.
- Create `src/kawaneen/acquisition/manifests.py` for lock, raw-file, snapshot, privacy-review, and record-manifest generation/validation.
- Create `src/kawaneen/acquisition/orchestrator.py` for plan, acquire/import, verify, audit, manifest, status, and rebuild workflows.
- Modify `src/kawaneen/cli.py` for `data` subcommands and denied-operation errors.
- Modify `pyproject.toml`, `uv.lock`, `.gitignore`, and `Makefile` for dependencies, ignored raw/private paths, and non-CI acquisition targets.
- Create `data/manifests/acquisition_specs/alarb.toml` and `data/manifests/acquisition_specs/arabiccr.toml`.
- Create the four requested version-controlled manifests with schema versions and deterministic ordering.
- Create `data/raw/README.md` and Phase 2 governance, privacy, demo, ADR, phase, and report documentation.
- Add offline fixture tests under `tests/test_acquisition_*.py`; mark live tests with `@pytest.mark.network` and exclude them by default.

## Task 1: Dependency and specification contract

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Create: `data/manifests/acquisition_specs/alarb.toml`, `data/manifests/acquisition_specs/arabiccr.toml`
- Create: `src/kawaneen/acquisition/__init__.py`, `src/kawaneen/acquisition/models.py`, `src/kawaneen/acquisition/specs.py`
- Test: `tests/test_acquisition_specs.py`

- [ ] Write failing tests for strict TOML loading, exact revisions/DOIs, expected row counts, split metadata, filename/format expectations, and denial of unknown fields.
- [ ] Run `uv run pytest tests/test_acquisition_specs.py -q` and confirm failure because the acquisition package and specs do not exist.
- [ ] Add `huggingface_hub` and `pyarrow` as runtime dependencies, create both TOML specifications, and implement typed Pydantic models and loader.
- [ ] Re-run the focused tests and confirm both specs load deterministically with repository-relative filenames and no raw payload content.

## Task 2: Registry-driven policy authorization

**Files:**
- Create: `src/kawaneen/acquisition/policy.py`
- Test: `tests/test_acquisition_policy.py`

- [ ] Write failing tests proving ALARB permits only evaluation/integrity/duplicate/privacy inspection, ArabiCCR permits only local research/inspection, and every other registry source is denied.
- [ ] Add tests proving training, publishing, public-display, and public-demo operations are denied and that no `force`, `bypass`, or unrestricted authorization parameter exists.
- [ ] Implement authorization from `load_registry()` plus source specifications and return typed denial reasons.
- [ ] Run focused policy tests and confirm all denied cases fail closed.

## Task 3: Secure immutable storage and adapters

**Files:**
- Create: `src/kawaneen/acquisition/storage.py`, `src/kawaneen/acquisition/adapters.py`
- Test: `tests/test_acquisition_storage.py`, `tests/test_acquisition_adapters.py`

- [ ] Write failing tests for traversal, absolute paths, symlink escapes, `.partial` cleanup, byte-preserving copies, SHA-256 while copying, and atomic destination replacement.
- [ ] Add a fixture-backed local-file adapter and tests for missing files, allowed source/version namespaces, and immutable existing raw files.
- [ ] Implement the official `huggingface_hub` adapter using the pinned full revision, selected files, cache download, temporary destination, hash-on-copy, validation callback, and atomic move.
- [ ] Investigate the current documented Mendeley public API. Implement an ArabiCCR API adapter only if a stable public download route is documented; otherwise expose only `import-local` and record the manual command.
- [ ] Add an opt-in `network` test marker without making live tests part of default pytest execution.

## Task 4: Integrity, schema, and duplicate analysis

**Files:**
- Create: `src/kawaneen/acquisition/integrity.py`
- Test: `tests/test_acquisition_integrity.py`, `tests/fixtures/acquisition/*.csv`, `tests/fixtures/acquisition/*.parquet`

- [ ] Write failing tests for empty files, checksum mismatch, unexpected filenames/formats, CSV BOM/UTF-8/header/row-count failures, corrupt Parquet, footer/schema/row-count mismatches, and deterministic schema fingerprints.
- [ ] Add exact duplicate tests for physical SHA-256 duplicates, canonical serialized row duplicates, ALARB train/test overlap, and reporting-only behavior.
- [ ] Implement streaming CSV validation and PyArrow Parquet validation without converting or mutating raw files.
- [ ] Run focused integrity tests and confirm reports contain counts only, never raw text.

## Task 5: Privacy screening and record manifests

**Files:**
- Create: `src/kawaneen/acquisition/privacy.py`, `src/kawaneen/acquisition/manifests.py`
- Test: `tests/test_acquisition_privacy.py`, `tests/test_acquisition_manifests.py`

- [ ] Write failing tests for deterministic masked emails, phones, IBAN-like values, context-supported identity numbers, passport/address indicators, identifier-like columns, and no raw examples in findings.
- [ ] Implement privacy findings with masked values and private bundle output only below ignored `artifacts/private/`; state that automated screening is not legal clearance.
- [ ] Implement deterministic lock, raw-file, snapshot, privacy-review, and local-record manifests with schema versions, repository-relative paths, sorted records, and atomic CLI-only writes.
- [ ] Generate record IDs from source ID, version, split, and row location; preserve ALARB official splits and explicitly record no ArabiCCR modelling split.
- [ ] Run focused privacy/manifest tests and inspect serialized output for stable ordering and absence of raw legal text.

## Task 6: Orchestration and CLI

**Files:**
- Create: `src/kawaneen/acquisition/orchestrator.py`
- Modify: `src/kawaneen/cli.py`
- Test: `tests/test_acquisition_cli.py`, `tests/test_acquisition_orchestrator.py`

- [ ] Write failing CLI tests for `data plan`, `data acquire`, `data import-local`, `data verify`, `data audit`, `data manifest build`, `data manifest validate`, `data status`, and `data rebuild`.
- [ ] Implement deterministic plan/status output, purpose validation, denied-source errors, import/acquire orchestration, verification/audit, manifest operations, and automatic rebuild without bypass options.
- [ ] Ensure acquisition is never invoked by `make check`, CI, or ordinary imports.
- [ ] Run all acquisition CLI tests offline and confirm denied sources never create raw files.

## Task 7: Make targets, ignore rules, and documentation

**Files:**
- Modify: `Makefile`, `.gitignore`
- Create: `data/raw/README.md`, requested Phase 2 governance/ADR/phase/report documents
- Test: existing CLI/help smoke tests and `git check-ignore` verification

- [ ] Add all requested data Make targets with documented underlying commands; keep network acquisition outside `check`.
- [ ] Ignore raw, partial, cache, private bundle, and acquisition artifact paths while retaining only sanitized manifests and documentation.
- [ ] Document authorization, immutable raw storage, integrity/privacy limitations, manual import fallback, public-demo prohibition, and Phase 3 eligibility rules.
- [ ] Record the live acquisition outcome, exact file metadata, hashes, schemas, duplicate counts, privacy findings, and manual review status without raw examples.

## Task 8: Full verification and staged handoff

- [ ] Run `uv sync --locked --dev`.
- [ ] Run all new CLI plan/policy commands and manifest validation.
- [ ] Run Ruff format/check, Pyright, pytest, pre-commit, `make check`, `git diff --check`, and ignored-path checks.
- [ ] Attempt live ALARB acquisition only after offline tests pass; remove any raw data and caches from the staged snapshot.
- [ ] Attempt ArabiCCR only through a verified documented public API; otherwise record the exact safe local-import command.
- [ ] Stage code, configuration, sanitized manifests, and documentation only; run `git diff --cached --check`.
- [ ] Confirm no commit or push occurs and report Phase 3 eligibility separately for ALARB, ArabiCCR, and the qualified MOJ-derived seed.

## Self-review checklist

- [ ] No blocked, metadata-only, conditional-unresolved, or excluded source has an acquisition path.
- [ ] No training, publishing, display, or demo authorization exists.
- [ ] Raw data and private artifacts are ignored and absent from the staged snapshot.
- [ ] All manifest paths are repository-relative and all generated ordering is deterministic.
- [ ] Privacy output is masked and explicitly not legal clearance.
- [ ] No parser, OCR, normalization, retrieval, model, API, or vector-database functionality is added.
