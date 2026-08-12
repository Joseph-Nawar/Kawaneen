# Phase 4 — Arabic normalization experiments

Phase 4 empirically compares three versioned, deterministic Arabic search-text
normalization policies over governed ALARB and ArabiCCR canonical units. The Phase 3
canonical `display_text` remains immutable. Normalized records are private derived
views with `display_text` preserved exactly and a policy-versioned `search_text`.

This phase does not chunk, embed, build production retrieval, create a Phase 6 human
evaluation set, or change source governance and permissions. Real-data queries, qrels,
per-example results, examples, indexes, and normalized Parquet remain ignored under
`artifacts/private/phase4_normalization/`. Only text-free aggregate manifests,
metrics, hashes, counts, and synthetic fixtures are version-controlled.

## Frozen experiment scope

The full eligible content-unit population contains 104,588 units: 53,364 ALARB units
and 51,224 ArabiCCR units. Indexing every local unit was not practical under the
available local resource limit, so a deterministic representative subset was frozen
before viewing retrieval results: 12,000 units, 6,000 from each source, with 1,500 in
each of the eight included content-bearing types. The included types are
`applicable_laws`, `case_text`, `court_reasoning`, `events`, `facts`, `reasoning`,
`ruling`, and `verdict`. Empty, whitespace-only, metadata, and OCR-derived units were
excluded. Candidate IDs and the selection manifest hash are recorded without source
text in `data/manifests/normalization/phase4_manifest.json`.

The same frozen candidate IDs, tokenizer, BM25 parameters, seed, challenge items, and
qrels were used for all three ablations. The challenge was generated from exact
canonical display text plus independent controlled perturbations; it was not produced
by applying any candidate normalizer.

## Product boundary

The selected product contract is `display_text + search_text`. Each private derived
record also retains the canonical/unit ID, document ID, unit type, policy ID/hash,
source and search hashes, ordinal, and provenance. Normalized text is never written
back to canonical Parquet or raw data.

The selected policy must be revalidated in Phase 7 against the later human evaluation
set. This 150-query Phase-4 challenge is diagnostic and must not be presented as that
future evaluation set.

The bounded final sensitivity validation preserves the frozen primary selection
`arabic-raw-v1` but recommends `arabic-light-v1` for downstream Phase-7 revalidation
because a separate 60-item short-query probe showed meaningful light-normalization
gains. This recommendation does not modify the primary challenge, its qrels, or the
primary manifest.
