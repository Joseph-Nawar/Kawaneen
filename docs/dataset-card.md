# Kawaneen dataset card

This card describes three deliberately separate data categories. The public
repository tracks metadata and aggregate evidence only; it does not reproduce
private legal text.

## A. Full/private/local canonical legal corpus

The full research corpus is a versioned local/private canonical layer built
from source categories recorded in the [source registry](data_sources.md),
including ALARB, ArabiCCR, and the named Saudi MOJ-derived seed. Statutory
reconciliation also records official Saudi authorities as manual reference
sources. These categories are not interchangeable: source terms, automated
access, privacy, original-source rights, and public-display permission are
assessed independently.

### Acquisition and processing

Acquisition is gated by the Phase 1 registry and source-specific TOML specs.
Raw files are copied byte-for-byte into ignored local storage with atomic
installation and SHA-256 manifests. Parsing/OCR routes document formats to
the documented backends, preserves source identity, and records health and
review metadata. Arabic normalization has explicit policy variants; canonical
units then feed structure-aware legal chunking with section/article provenance.
See [data acquisition](data-acquisition.md), [governance](data-governance.md),
[canonical corpus](canonical-corpus.md), and [parsing/OCR](parsing-and-ocr.md)
for the authoritative operational record.

The frozen serving corpus identity is
`290d7a91e5f435778e782b76284a9797fb7f5ae261380f0a923b56224e530daa`; the
Phase 8 manifest reports 26,147 documents. That hash identifies a private
local snapshot, not a redistribution package.

### Rights and governance

The full corpus is not redistributed. ALARB and ArabiCCR are limited to their
recorded local research/evaluation purposes; the MOJ-derived seed is not
evaluation-approved or public-display-approved. Official-text copyright status
does not grant permission to scrape, mirror, quote publicly, or bypass portal
terms. No training or public-demo operation is authorized by the current
source policy.

### Known gaps and bias

The Phase 8 metadata audit reports 0% population for six structured filter
fields across 26,147 documents, so filters exist but have no metadata
leaderboard result. The corpus and evaluation strata may not represent all
Arabic legal domains, jurisdictions, dialects, document formats, or OCR
conditions. Search metrics are not legal-correctness metrics.

## B. Frozen DEV/HOLDOUT evaluation assets

Evaluation assets are frozen, provenance-hashed releases used to measure
retrieval, grounding, generation policy, extraction behavior, and diagnostics.
DEV is the permissible surface for the recorded Phase 15 engineering analyses.
HOLDOUT is protected, frozen before its one recorded evaluation, and must not
be accessed, tuned on, or retuned during this portfolio closeout.

Phase 6’s public metadata describes 200 base intents and 40 variants with
`human_verified=false` and independent AI source review. Phase 15 adds named
DEV-only populations, including 141 embedding queries, a corrected 46-query
hard reranking slice, 40 citation candidates, and an 80-case matched generator
population. The 30-case diagnostic is enriched operational analysis, not human
gold, expert review, or a prevalence estimate.

Tracked manifests and reports expose hashes, counts, provenance, and aggregate
results. Query text, qrels, evidence spans, raw generation, per-example
records, and private review packets remain ignored/local. Labels are useful for
engineering diagnostics but are not an independent human legal-expert ground
truth. See [evaluation artifacts](../data/evaluation/README.md) and the
[Phase 15 report](reports/phase-15-evaluation-and-experiment-report.md).

## C. Public synthetic `KAWANEEN_DEMO` corpus

The public demo contains approximately 64 project-created Arabic passages in
the fictional jurisdiction `KAWANEEN_DEMO`. They are synthetic statutes and
fact patterns, not Saudi law, official legislation, or legal advice. The
corpus is designed for a safe product walkthrough and deterministic retrieval/
abstention behavior, not for legal model training or correctness claims.

The public profile bundles precomputed vectors for
`intfloat/multilingual-e5-small` at revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`, uses BM25 plus exact NumPy dense
search and optional top-4 reranking, and returns exact synthetic evidence
passages. No private corpus, Qwen/Ollama, Qdrant, MLflow, or Saudi source text
is included. The bundle is intended for redistribution as the project-created
demo material under the repository’s MIT licence; third-party model terms and
runtime download terms remain governed by their own licenses and hosting
conditions.

The demo is bounded by the public profile’s input, evidence, extraction,
concurrency, and rate limits. Publication to Hugging Face has not occurred and
is optional after this closeout.
