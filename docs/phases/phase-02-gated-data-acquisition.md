# Phase 2 — Gated Data Acquisition, Integrity Validation, and Privacy Inspection

## Scope

Phase 2 establishes source-specific authorization, controlled acquisition/import, immutable raw storage, deterministic integrity and duplicate analysis, and automated privacy screening. It does not parse, OCR, normalize, retrieve, train, serve, publish, or create a public demo.

## Exit criteria

- ALARB, ArabiCCR, and the qualified Saudi MOJ-derived seed are represented by pinned, version-controlled specifications.
- Only authorized local purposes can acquire or inspect them; all other registry sources are denied.
- Raw storage rejects unsafe paths and preserves original bytes.
- Required manifests, checksums, schemas, row counts, split overlap, duplicate counts, and privacy statuses are deterministic.
- Tests remain offline by default; live tests, if added later, are explicitly marked `network`.
- No source is eligible for Phase 3 until its recorded rights and privacy review permit the proposed use. The MOJ-derived seed is currently eligible only as a private raw parsing seed, not as an authoritative or gold article corpus.

The Phase 2 report records the actual acquisition attempt and any manual action required. A failed or unavailable source remains blocked; it is not treated as successfully acquired.
