# Canonical corpus

Phase 3 creates deterministic, provenance-complete local views of the Phase 2 source
artifacts. It does not change raw bytes, normalize Arabic, clean text, chunk for
retrieval, train models, or authorize public display.

Each canonical document and unit carries the source ID, exact source version, artifact
path, one-based source row, source field, and official split where applicable. IDs are
UUIDv5 values derived from those locations, never from raw text. Rebuilding the same
inputs therefore produces stable IDs and deterministic Parquet metadata.

ALARB maps one case per source row and preserves train/test provenance. ArabiCCR maps
one case per row and keeps its judgment, court, type, date, year, city, and URL fields
as source metadata; `case_text`, `EVENTS`, `REASONING`, and `RULING` remain explicitly
source-derived units. The MOJ seed first creates one immutable fragment per row, then
creates conservative local article candidates without silently merging ambiguous keys.

Canonicalization inherits Phase 2 restrictions. ALARB and ArabiCCR remain private
local parsing/evaluation material, with training and public display denied. The MOJ
seed is a private raw parsing seed only and is not an authoritative or gold corpus.
