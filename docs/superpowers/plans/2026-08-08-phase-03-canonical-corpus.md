# Phase 3 Canonical Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build deterministic, provenance-complete canonical Parquet views of the three acquired Phase 2 sources and add a separately optional PDF/OCR routing and benchmark subsystem without changing raw text, normalizing Arabic, or enabling retrieval or modelling.

**Architecture:** `kawaneen.corpus` owns canonical envelopes, typed statute/case documents, source adapters, conservative statutory fragment reconstruction, reconciliation/gap metadata, deterministic Parquet output, and sanitized manifests. `kawaneen.parsing` is an optional boundary: it probes PDF text health, chooses a configured route, and exposes Docling/RapidOCR adapters lazily; it is never invoked for CSV/Parquet corpus sources. Raw Phase 2 files remain read-only and ignored, while canonical/interim and private benchmark outputs are ignored.

**Tech Stack:** Python 3.11–3.12, Pydantic, PyArrow, stdlib `uuid`, CSV/JSON/TOML, optional `pypdf`, Docling, and RapidOCR. No PyMuPDF, normalization library, retrieval stack, embeddings, models, or network-dependent tests.

## Global Constraints

- Do not modify Phase 2 raw files.
- Do not perform Arabic normalization, digit replacement, diacritic removal, stemming, cleaning, fuzzy deduplication, retrieval chunking, embeddings, RAG, training, public display, or APIs.
- Preserve every source text value exactly as loaded and retain raw labels beside derived structural metadata.
- Every raw record must produce an accounted status and every canonical unit must contain exact source/version/file/row/field provenance.
- Canonical outputs are typed Parquet under ignored `data/interim/canonical/<source>/<version>/`.
- Committed manifests contain only sanitized counts, identities, hashes, statuses, and repository-relative paths.
- ALARB and ArabiCCR inherit Phase 2 privacy and use restrictions; canonicalization does not grant training or public-display rights.
- The MOJ-derived seed remains a private parsing seed and is not an authoritative or gold corpus before reconciliation.
- Official authorities are manual reconciliation references only; no bulk scraping.
- Ordinary `make check`, CI, and default tests remain offline and do not require raw data, private PDFs, OCR models, or optional parsing dependencies.

## File Map

- Create `src/kawaneen/corpus/models.py` for typed envelopes, documents, units, fragments, reconciliation records, and build results.
- Create `src/kawaneen/corpus/ids.py` for the fixed UUIDv5 namespace and deterministic document/unit/fragment IDs.
- Create `src/kawaneen/corpus/serialization.py` for typed Arrow schemas, Parquet writes, hashes, and safe interim paths.
- Create `src/kawaneen/corpus/adapters.py` for ALARB, ArabiCCR, and Saudi MOJ-derived mappings.
- Create `src/kawaneen/corpus/statutory.py` for article-label parsing, duplicate-group classification, conservative reconstruction, reconciliation fields, inventory, and gap reports.
- Create `src/kawaneen/corpus/orchestrator.py` for plan, build, validate, inventory, statutory status, and gap workflows.
- Create `src/kawaneen/corpus/__init__.py` with the public corpus API.
- Create `src/kawaneen/parsing/models.py`, `health.py`, `routing.py`, `docling_backend.py`, and `benchmark.py` for optional PDF health/routing and offline metric calculations.
- Modify `src/kawaneen/cli.py` for `corpus plan`, `build`, `validate`, `inventory`, `statutory-status`, and `gaps`.
- Modify `pyproject.toml`, `uv.lock`, `.gitignore`, and `Makefile` for optional parsing dependencies, ignored outputs, and non-CI corpus targets.
- Create `configs/parsing/default.toml` with explicit thresholds and route settings.
- Extend `data/manifests/reconciliation/core-commercial-civil-v1.csv` with typed reconciliation and authority fields.
- Create sanitized canonical manifests under `data/manifests/canonical/` and `statutory_gap_report.csv`.
- Create `data/interim/canonical/README.md`, `data/benchmarks/README.md`, and private ignored benchmark directories.
- Create/update Phase 3 governance, architecture, parser, ADR, phase, and report documentation.
- Add focused offline tests for models, IDs, mappings, reconstruction, serialization, CLI, parser routing, and metrics.

## Task 1: Canonical contracts and deterministic IDs

**Files:**
- Create: `src/kawaneen/corpus/models.py`, `src/kawaneen/corpus/ids.py`, `src/kawaneen/corpus/__init__.py`
- Test: `tests/test_corpus_models.py`, `tests/test_corpus_ids.py`

- [ ] Write failing tests for a shared provenance envelope, discriminated `statute`/`case` documents, typed units, immutable source fragments, required provenance, and stable UUIDv5 IDs.
- [ ] Run `uv run pytest tests/test_corpus_models.py tests/test_corpus_ids.py -q`; confirm failure because the corpus package does not exist.
- [ ] Implement frozen Pydantic models with `kind: Literal[...]`, `source_id`, `source_version`, `source_path`, `source_row`, `source_field`, `split`, raw labels, and use-stage restrictions.
- [ ] Implement IDs as UUIDv5 over stable strings containing the fixed namespace, source ID, version, source artifact path, row location, unit type, and field; never hash raw text.
- [ ] Re-run focused tests and verify repeated construction yields byte-identical IDs.

## Task 2: Typed Parquet serialization and safe output

**Files:**
- Create: `src/kawaneen/corpus/serialization.py`
- Modify: `.gitignore`
- Test: `tests/test_corpus_serialization.py`

- [ ] Write failing tests for typed `documents.parquet`, `units.parquet`, and `fragments.parquet`, deterministic row ordering, output hashes, relative paths, and refusal to write below raw data.
- [ ] Implement explicit PyArrow schemas, atomic `.partial` writes, deterministic metadata ordering, and SHA-256 summaries without embedding raw text in committed manifests.
- [ ] Add ignored `data/interim/canonical/`, private parsing benchmark, OCR cache/model, and canonical output paths while retaining README metadata.
- [ ] Re-run serialization tests and assert identical input/output produces identical bytes and hashes.

## Task 3: Source adapters and accounting

**Files:**
- Create: `src/kawaneen/corpus/adapters.py`
- Test: `tests/test_corpus_adapters.py`

- [ ] Write failing fixture tests for one ALARB train/test row, one ArabiCCR row with all source-derived fields, and one MOJ statute row; assert exact source text preservation and complete field provenance.
- [ ] Implement adapters reading only the existing Phase 2 raw namespaces and emitting canonical documents/units/fragments; never alter or rewrite raw bytes.
- [ ] Preserve ALARB official splits and map facts, reasoning, applicable laws, and verdict to typed units.
- [ ] Preserve ArabiCCR case metadata and map `case_text`, `EVENTS`, `REASONING`, and `RULING` as source-derived units without claiming independent authority.
- [ ] Emit one immutable MOJ `SourceFragment` per source row before any grouping.
- [ ] Add raw-record accounting with accounted, canonicalized, and excluded/error counts; no silent drops.
- [ ] Run focused adapter tests and verify Phase 2 source hashes before and after a fixture-backed build.

## Task 4: Statutory identifiers and conservative reconstruction

**Files:**
- Create: `src/kawaneen/corpus/statutory.py`
- Test: `tests/test_corpus_statutory.py`

- [ ] Write failing tests for Western digits, Arabic-Indic digits, Arabic ordinal wording, explicit part indicators, unique groups, explicit fragment series, ambiguous duplicate refusal, conflicting duplicates, and unresolved groups.
- [ ] Implement raw-label-preserving article parsing that emits derived ordinals/parts only as structural metadata.
- [ ] Group fragments by law name and untouched article label; classify every duplicate group as one controlled status and preserve all fragment IDs and ordering evidence.
- [ ] Merge only structurally explicit fragment series; keep ambiguous/conflicting groups as separate article candidates with reconstruction operations recorded.
- [ ] Compute sanitized reconstruction counts including all 1,993 MOJ duplicate-key situations and never put text in reports.
- [ ] Run focused statutory tests and confirm no raw text appears in serialized summary output.

## Task 5: Reconciliation, inventory, and gap analysis

**Files:**
- Modify: `data/manifests/reconciliation/core-commercial-civil-v1.csv`
- Create: `data/manifests/canonical/inventory.json`, `data/manifests/canonical/quality.json`, `data/manifests/canonical/reconstruction.json`, `data/manifests/canonical/snapshot.json`, `data/manifests/canonical/statutory_gap_report.csv`
- Test: `tests/test_corpus_reconciliation.py`, `tests/test_corpus_manifests.py`

- [ ] Write failing tests for typed reconciliation fields, explicit manual-review states, recognized authorities, deterministic inventory rows, gap statuses, and no absolute paths/raw text.
- [ ] Extend the reconciliation template with official title/source URL/status/publication/effective/amendment dates, dataset/official counts, reviewed samples, discrepancies, authority, reconciliation status, and v1 eligibility.
- [ ] Generate a sanitized inventory for all three sources and a 15–25 instrument gap report identifying targeted needs without acquiring missing laws.
- [ ] Record manual authority options as MOJ Legal Portal, Bureau of Experts, and Saudi National Platform; leave unavailable values as `not_reviewed` or `manual_review_required`, never guessed.
- [ ] Run focused manifest tests and verify deterministic rebuilds and repository-relative paths.

## Task 6: Corpus orchestration, CLI, and Make targets

**Files:**
- Create: `src/kawaneen/corpus/orchestrator.py`
- Modify: `src/kawaneen/cli.py`, `Makefile`
- Test: `tests/test_corpus_cli.py`, `tests/test_corpus_orchestrator.py`

- [ ] Write failing tests for `corpus plan`, build with source selection, validate, inventory, statutory-status, and gaps, including missing raw-data behavior.
- [ ] Implement deterministic offline plan/status output and real-data build orchestration outside ordinary checks.
- [ ] Add Make targets for corpus plan/build/validate/inventory/statutory-status/gaps; keep builds out of `make check`.
- [ ] Run focused CLI tests and verify no CLI operation offers a raw-file mutation or rights bypass.

## Task 7: Optional PDF health, Docling routing, OCR abstraction, and metrics

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Create: `configs/parsing/default.toml`, `src/kawaneen/parsing/models.py`, `health.py`, `routing.py`, `docling_backend.py`, `benchmark.py`
- Test: `tests/test_parsing_routing.py`, `tests/test_parsing_metrics.py`

- [ ] Write failing tests for healthy embedded text, damaged/mixed text, image-only scans, configured thresholds, fake parser/OCR provenance, CER/WER, heading metrics, article-number accuracy, reading order, and page-reference preservation.
- [ ] Add an optional `parsing` dependency group with `pypdf`, Docling, and the selected RapidOCR runtime; do not import them at package import time or download models in CI.
- [ ] Implement a lazy pypdf health probe, route model, Docling adapter boundary, explicit OCR engine/model provenance, and fake backend tests; do not add PyMuPDF.
- [ ] Implement deterministic benchmark metric functions and project gate evaluation, including critical article-number error detection.
- [ ] Run focused parser tests without optional dependencies and document unavailable local benchmark data.

## Task 8: Documentation, real-corpus build, and verification

**Files:**
- Create: `docs/canonical-corpus.md`, `docs/statutory-reconciliation.md`, `docs/parsing-and-ocr.md`, `docs/adr/0006-canonical-corpus-and-parser-routing.md`, `docs/phases/phase-03-canonical-corpus.md`, `docs/reports/phase-03-canonical-corpus-report.md`, `data/interim/canonical/README.md`, `data/benchmarks/README.md`
- Test/verify: full repository suite

- [ ] Document canonical schemas, exact-text guarantees, provenance, source restrictions, reconstruction limits, manual authority boundaries, parser routes, dependency licences, benchmark gates, and Phase 4 eligibility.
- [ ] Build all three acquired sources locally, record output hashes/counts/statuses, and verify all Phase 2 raw hashes remain unchanged.
- [ ] Run corpus plan/build/validate/inventory/status/gaps, parser probe/benchmark commands where private data exists, Ruff, Pyright, pytest, pre-commit, `make check`, and both Git diff checks.
- [ ] Confirm raw data, canonical interim outputs, private PDFs, OCR caches/models, and privacy bundles are ignored; stage only code, configs, sanitized manifests, tests, and docs.
- [ ] Report remaining legal/manual reconciliation actions and separately state Phase 4 eligibility for each source.
