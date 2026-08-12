# Phase 5 — Legal Structure and Chunking

Phase 5 builds deterministic citation-safe chunks from immutable Phase 3
canonical units. It uses the Phase 4 `arabic-light-v1` policy only after exact
canonical boundaries and source spans are finalized. No embeddings, dense or
hybrid retrieval, reranking, RAG, Qdrant, APIs, or Phase 6 human evaluation are
included.

## Source hierarchy

- ALARB: `document → facts/court_reasoning/applicable_laws/verdict → paragraph/clause`.
- ArabiCCR: `document → EVENTS/REASONING/RULING → paragraph/clause` when those
  structured children are populated. `case_text` is fallback document content
  only when structured children are unavailable; it is not blindly indexed as a
  duplicate.
- No chapter, page, section, or article information is inferred without source
  evidence. The current Phase 3 judgment units provide canonical section-level
  evidence, so deeper leaves are conservative paragraph/sentence spans.

## Chunk contract

Every chunk stores a deterministic ID, source/document IDs, exact canonical
source spans, immutable `display_text`, normalized `search_text`, structure and
sibling links, citation anchor, token count, both policy IDs/hashes, and
provenance. Fixed windows may cross structural boundaries but retain all
contributing source spans. Structural, neighbor, and parent-child chunks remain
within the same section parent.

Structural splitting uses existing boundaries, paragraph/list/clause breaks,
conservative sentence boundaries, then a 512-token window with 64-token overlap
for an oversized sentence. The target is 384 tokens and the normal maximum is
512. The tokenizer is the Phase 4 Unicode tokenizer and performs no hidden Arabic
normalization.

## Strategies

`fixed-256-v1` uses 256-token windows with 32-token overlap; `fixed-512-v1`
uses 512/64. `legal-structure-v1` uses boundary-aware leaves;
`legal-structure-neighbor-v1` adds only previous/current/next sibling context
from the same parent; `legal-parent-child-v1` indexes children and aggregates
parent ranking by deterministic maximum child score while retaining the best
child citation.

The selected strategy and aggregate evidence are recorded in
`data/manifests/chunking/phase5_chunking_manifest.json` and
`data/evaluation/phase5_chunking_metrics.json`. Real chunks, challenge queries,
source spans, qrels, and per-example results remain private under
`artifacts/private/phase5_chunking/`.

Article-level retrieval must be revalidated when a trusted authoritative
statutory corpus becomes available. This diagnostic challenge is not the Phase 6
human evaluation set.
