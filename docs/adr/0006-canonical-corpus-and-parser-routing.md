# ADR 0006: Canonical corpus and parser routing

## Status

Accepted for Phase 3 implementation.

## Decision

Use deterministic UUIDv5 canonical identifiers and typed Parquet documents/units with
exact source provenance. Keep source adapters separate from statutory reconstruction.
Represent every MOJ row as an immutable fragment and merge only explicit fragment
series. Keep PDF health probing, Docling layout parsing, OCR routing, and benchmark
metrics in an optional package boundary.

## Consequences

Canonical outputs are reproducible local derivatives, not a licence grant or public
corpus. Raw text remains untouched and every unit can be audited back to its source
artifact. Ambiguous statutory duplicates remain visible instead of being silently
collapsed. Optional parser dependencies do not burden the foundation install or CI.
