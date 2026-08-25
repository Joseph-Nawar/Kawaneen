# Phase 11A v2 Annotation Repair Implementation Plan

> **For agentic workers:** Execute this plan inline with focused verification checkpoints. Do not run model inference, HOLDOUT label evaluation, full pytest, coverage, commit, or push.

**Goal:** Replace the unsuitable Phase 11A v1 selection with a reproducible 120-unit structurally atomic regulatory benchmark and repair deterministic temporal/regulation candidate detection before annotation begins.

**Architecture:** Keep the extraction schema and Phase 10 checkpoint lifecycle unchanged. Add a versioned Phase-11 unit-eligibility policy and selection/candidate fingerprints; select only canonical article/clause/paragraph-scale units within 50–1500 characters; keep private source-bearing v2 batch/audit artifacts under `artifacts/private/`.

**Tech Stack:** Python 3.12, Pydantic, PyArrow, uv, pytest focused tests, Ruff, Pyright.

**Spec:** User request for bounded Phase-11A annotation-set and deterministic-candidate repair.

## Global Constraints

- Use the governed `saudi-moj-derived` regulatory universe and source-role/type metadata; no source-name hacks.
- Preserve TOTAL 120 / DEV 80 / HOLDOUT 40 / SMOKE 10 and document-disjoint DEV/HOLDOUT selection.
- Use weak cues only for stratification; never create semantic labels.
- Preserve exact Arabic text/codepoint offsets and raw candidate text.
- Do not load or evaluate HOLDOUT annotation content.
- Do not run Qwen/Ollama, hybrid inference, full pytest, coverage, commit, or push.
- Keep all source-bearing v2 artifacts private and tracked artifacts text-free.

### Task 1: Audit the canonical structural universe

**Files:**
- Inspect: `src/kawaneen/corpus/models.py`, `src/kawaneen/extraction/annotation.py`, `src/kawaneen/extraction/candidates.py`
- Inspect: `data/interim/canonical/**/units.parquet`, source registry metadata, v1 manifests

- [ ] Report unit-type/structural-role, length, document, and fallback aggregates without printing source text.
- [ ] Identify actual article/clause/paragraph-scale metadata and confirm the fixed 50–1500 character bound is usable.

### Task 2: Add failing focused tests

**Files:**
- Create/modify: `tests/test_extraction_unit_eligibility.py`
- Create/modify: `tests/test_extraction_candidates.py`
- Create/modify: `tests/test_extraction_v2_regeneration.py`

- [ ] Test atomic eligibility, hard length boundaries, fallback exclusion, and document-disjoint reselection.
- [ ] Test Arabic/ASCII numeric and spelled-duration candidates, exact offsets, regulation positive/negative patterns, and money/percentage regressions.
- [ ] Test v2 fingerprint incompatibility, reset annotation state, batch counts, and privacy.
- [ ] Run only these focused tests and confirm they fail for the missing policy/repairs.

### Task 3: Implement policy and deterministic repairs

**Files:**
- Modify: `src/kawaneen/extraction/annotation.py`
- Modify: `src/kawaneen/extraction/candidates.py`
- Modify: `src/kawaneen/extraction/normalization.py`
- Modify: `src/kawaneen/extraction/orchestration.py`
- Modify/add: `src/kawaneen/extraction/selection.py` or the smallest existing selection boundary

- [ ] Add a versioned `phase11-eligibility-v2` policy using governed source-role/type, structural role, and 50–1500 character bounds.
- [ ] Exclude fallback/document/full-text/header/aggregate units without ad hoc splitting.
- [ ] Add conservative Arabic duration recognition with numeric normalization only when unambiguous; retain dates as separate temporal candidates.
- [ ] Tighten regulation references to bounded named-instrument/reference patterns and retain exact spans.
- [ ] Preserve existing money/percentage behavior unless focused regression tests reveal a real defect.
- [ ] Make v1 selection/checkpoint fingerprints incompatible with v2.

### Task 4: Regenerate private v2 artifacts

**Files:**
- Generate privately: `artifacts/private/phase11_extraction/review/phase11_dev_annotation_batch_v2.json`
- Generate privately: v2 annotation records, deterministic candidate audit sample, and superseded-v1 audit metadata
- Replace tracked: v2 selection/readiness manifests only, with no source text

- [ ] Regenerate exactly 120/80/40/10 with deterministic corpus/policy hashes, document-disjoint splits, negative/low-signal strata, and all records reset to unreviewed/unreviewed/false/null.
- [ ] Export exactly 80 DEV / 0 HOLDOUT with the existing annotation contract and corrected candidates.
- [ ] Write a private candidate-audit sample of up to 20 temporal/regulation and bounded money/percentage candidates with private context.

### Task 5: Verify and report

- [ ] Run focused tests, Ruff, targeted Pyright, privacy audit, v2 hash/count checks, DEV progress/validator, and no model/HOLDOUT checks.
- [ ] Confirm max canonical text length ≤1500 and DEV/HOLDOUT document overlap = 0.
- [ ] Report root causes, policy, detector changes, v2 distributions, paths, hashes, and explicit safety confirmations.
