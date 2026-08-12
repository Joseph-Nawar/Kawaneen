# Phase 5 legal structure and chunking report

Date: 2026-08-12

## Scope and freeze

The experiment froze 1,500 whole ALARB documents and 1,500 whole ArabiCCR
documents before evaluation, retaining all selected canonical children: 3,000
documents and 12,000 canonical units. The source scope hash and document-ID hash
are in the tracked corpus manifest. Only ALARB and ArabiCCR were used; no
OCR-derived content or MOJ text was used as retrieval gold. ArabiCCR `case_text`
was excluded from indexing when EVENTS/REASONING/RULING structured children were
available, avoiding duplicate document content.

The private challenge is `phase5-chunk-challenge-v1`, seed `20260812`, with 180
source-span-based items: six slices × 30. The slices are local passage, long legal
section, multi-paragraph evidence, structural-boundary proximity, fixed-window
boundary stress, and parent-context evidence. There are 34 multi-relevant qrels
for duplicate/parallel evidence; the largest duplicate group has 117 relevant
spans. Queries and gold spans were built from canonical text before strategy
construction and are stored only under the ignored Phase 5 private root.

## Policy hashes and chunk statistics

| Strategy | Chunks | Mean tokens | Median | P95 | Max | Indexed tokens | Duplication | Fallbacks | Source coverage | Fixed boundary crossings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed-256-v1 | 16,135 | 235.28 | 256 | 256 | 256 | 3,796,235 | 1.1245 | 0 | 100% | 7,185 |
| fixed-512-v1 | 8,609 | 433.84 | 512 | 512 | 512 | 3,734,891 | 1.1063 | 0 | 100% | 5,622 |
| legal-structure-v1 | 14,164 | 239.74 | 231 | 479 | 512 | 3,395,691 | 1.0059 | 550 | 100% | 0 |
| legal-structure-neighbor-v1 | 14,164 | 239.74 | 231 | 479 | 512 | 3,395,691 | 1.0059 | 550 | 100% | 0 |
| legal-parent-child-v1 | 14,164 | 239.74 | 231 | 479 | 512 | 3,395,691 | 1.0059 | 550 | 100% | 0 |

All strategies had zero orphan, cycle, invalid-span, and structural cross-parent
violations. Fixed-window crossings are expected baseline behavior and are reported
separately as boundary stress, not treated as structural integrity failures.

Policy hashes: fixed-256 `caf5cf8e…b3691323`; fixed-512
`2735ee4a…e969add`; structure `93fe26c1…acca46f`; neighbor
`df377a60…ba4ba704`; parent-child `69036233…f43f3bda`.

## Retrieval ablation

The same tokenizer, `arabic-light-v1`, BM25 parameters (`k1=1.2`, `b=0.75`),
candidate scope, query set, qrels, seed, and top-10 cutoff were used for every
strategy.

| Strategy | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| fixed-256-v1 | 0.4844 | 0.6624 | 0.7278 | 0.6460 | 0.6394 |
| fixed-512-v1 | 0.4807 | 0.6622 | 0.7085 | 0.6118 | 0.6227 |
| legal-structure-v1 | 0.5884 | 0.6836 | 0.7392 | 0.6916 | 0.6837 |
| legal-structure-neighbor-v1 | 0.5606 | 0.6832 | 0.7365 | 0.6800 | 0.6781 |
| legal-parent-child-v1 | 0.5884 | 0.6864 | 0.7336 | 0.6913 | 0.6824 |

`fixed-256-v1` was the best fixed baseline. The structural strategy was within
0.02 of that baseline on Recall@10 and exceeded it on MRR@10, while improving
citation quality. Neighbor and parent-child behavior did not meet the separate
context-improvement promotion threshold; neighbor attribution leakage was 3.33%.

## Citation and context metrics

| Strategy | Span cov@1 | Span cov@5 | Citation precision@1 | Citation recall@1 | Anchor accuracy@1 | Overreach@1 | Multi-structure citation |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-256-v1 | 0.5722 | 0.7278 | 0.0363 | 0.5290 | 0.5722 | 0.9637 | 0.7000 |
| fixed-512-v1 | 0.5167 | 0.7278 | 0.0190 | 0.5073 | 0.5167 | 0.9810 | 0.8889 |
| legal-structure-v1 | 0.6444 | 0.7444 | 0.1299 | 0.5988 | 0.6444 | 0.8701 | 0.0000 |
| legal-structure-neighbor-v1 | 0.6167 | 0.7556 | 0.1267 | 0.5736 | 0.6167 | 0.8733 | 0.0000 |
| legal-parent-child-v1 | 0.6444 | 0.7444 | 0.1299 | 0.5988 | 0.6444 | 0.8701 | 0.0000 |

Structure improved citation precision over the best fixed baseline by 0.0936
absolute and anchor accuracy by 0.0722. Neighbor context coverage@5 was 0.7667
versus 0.7444 for structure, but the gain was below the predefined +0.05
promotion rule and included attribution leakage. Parent-child coverage@1 was
0.6444 with deterministic max-child aggregation, but it did not clearly improve
over the structural leaf policy.

## Paired comparisons

The selected structure versus fixed-256 paired deltas (structure minus fixed-256)
were:

- MRR@10: +0.0456, deterministic bootstrap 95% CI `[0.0041, 0.0904]`.
- nDCG@10: +0.0442, CI `[0.0094, 0.0819]`.
- Recall@10: +0.0114, CI `[-0.0314, 0.0601]`.

Query-level MRR wins/ties/losses were 36/124/20 for structure versus fixed-256.
Parent-child versus structure was effectively tied at 1/178/1, with intervals
crossing zero. Neighbor versus structure was 16/148/16, also without a consistent
positive paired interval.

## Decision and limitations

The selected downstream chunk policy is `legal-structure-v1`: it passed all hard
integrity gates, stayed within the fixed-baseline retrieval tolerance, and
materially improved citation precision and structural-anchor accuracy. Neighbor
and parent-child policies remain experimental and were not promoted.

The challenge is diagnostic, not Phase 6 gold. The corpus contains canonical
judgment sections rather than a trusted authoritative statutory retrieval corpus;
article-level retrieval must be revalidated when that authoritative source is
available. The selected policy also requires later evaluation against the Phase 6
human evaluation set. No production retrieval system, embeddings, dense/hybrid
retrieval, reranking, RAG, or API work was started.
