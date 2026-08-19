# Phase 7 — Retrieval Baselines

Phase 7 defines reproducible keyword, Okapi BM25, and exact dense retrieval baselines over the frozen `phase6-retrieval-eval-ai-reviewed-v1` release. It uses the existing Phase 4 tokenizer/normalizers, Phase 5 `legal-structure-v1` chunk boundaries, and Phase 6 evidence-derived chunk qrels without changing any earlier artifact.

The tracked experiment outputs are text-free. Query-level rankings, failure packets, embeddings, and derived corpus chunks remain private below `artifacts/private/phase7_retrieval/`.

## Fixed contract

- Keyword: deterministic token-set Jaccard, raw/light normalization, `chunk_id` tie-break.
- BM25: `bm25s`, Okapi parameters `k1=1.2`, `b=0.75`, raw/light normalization.
- Dense: `intfloat/multilingual-e5-small` and `BAAI/bge-m3`, normalized float32 vectors, exact NumPy or `faiss.IndexFlatIP` search, no sparse/ColBERT BGE outputs.
- E5 formatting: `query: {text}` and `passage: {text}`; BGE has no E5 prefix.
- Dense length contract: E5 document/query `max_length=512`; BGE-M3 document/query `max_length=1536`.
- Dev selection is based on nDCG@10 with the predeclared 0.005 threshold and a material Recall@10 guard. Holdout consumes the frozen selection only.
- Bootstrap comparisons use 2,000 paired replicates, seed `20260815`, and 95% percentile intervals.

## Execution

```text
kawaneen retrieval plan
kawaneen retrieval build-corpus
kawaneen retrieval smoke
kawaneen retrieval encode-corpus --model bge-m3 --resume
kawaneen retrieval cache-status --model bge-m3
kawaneen retrieval real-model-smoke  # explicit local model-loading check
kawaneen retrieval evaluate-dev
kawaneen retrieval freeze-dev-selection
kawaneen retrieval evaluate-holdout --allow-holdout
kawaneen retrieval report
```

The final tracked manifests and metrics are written under `data/manifests/retrieval/`
and `data/evaluation/`; rankings, failure packets, derived chunks, and embedding caches
remain private under `artifacts/private/phase7_retrieval/`.

Dense corpus encoding is resumable in deterministic 1,024-chunk checkpoint blocks. The
standalone BGE command defaults to the resolved safe CPU device. E5 MPS was usable only
on the eventual successful path; BGE-M3 MPS model transfer and Metal synchronization
were unreliable, so no automatic MPS/CPU switching is performed for BGE.

The BGE command reports tokenizer-only corpus and evaluation-query length diagnostics
before inference. E5's 512-token truncation remains part of its model contract. BGE-M3
uses 1,536 tokens independently of E5; the measured Phase-7 corpus maximum is 1,212
BGE tokens, so the current corpus has an expected BGE truncation count of 0/124,311.
This does not change the frozen Phase 5 chunk boundaries, corpus, queries, or qrels.

Normal tests use synthetic corpora and mocked dense encoders. Real model loading is an explicit local experiment concern and never occurs during CI tests.

## Scope boundary

This phase intentionally stops at independently measured lexical and dense baselines. Hybrid fusion, reciprocal-rank fusion, learned fusion, reranking, generation, RAG, and fine-tuning are Phase 8 or later concerns.
