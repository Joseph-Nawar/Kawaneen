# Phase 8 — Hybrid Retrieval and Reranking

Phase 8 adds a deliberately fixed hybrid candidate stage around the frozen
Phase-7 BM25 and BGE retrieval universe. The DEV-only reranker evaluation is
complete; holdout remains protected until the frozen DEV selection is reviewed
and explicitly executed.

## Motivation and protocol

Phase 7 showed BM25 as the dominant DEV baseline: its Arabic-light policy,
Okapi parameters (`k1=1.2`, `b=0.75`), and exact qrel/slice evaluation remain
frozen. BGE-M3 is retained because its complementary dense recall can rescue
some lexical misses even when its aggregate ranking is weaker. RRF therefore
combines the fixed top-50 lists with `rrf_k=60` and emits 20 candidates.

The only tested fusion ladder is equal RRF and sparse weight 1.0 with dense
weights 0.25, 0.50, 0.75, and 1.00. BM25 and BGE Phase-7 baselines are reported
alongside it. Selection uses DEV nDCG@10, subject to no more than 0.01 absolute
regression in Recall@10 or CompleteEvidenceRecall@10 versus BM25. Differences
under 0.002 nDCG@10 prefer the lower dense weight. This is a provisional
fusion choice only; no overall Phase-8 selection or holdout manifest is
created before reranker evaluation.

CandidateRecall@20 and CandidateCompleteEvidenceRecall@20 are the candidate
stage ceiling diagnostics. Every query retains source ranks, raw scores,
fused score/rank, and sparse-only/dense-only/both provenance in private
artifacts. Tracked metrics contain IDs only, not query or passage text.

## Reranker architecture and cost

The later reranker adapter targets `BAAI/bge-reranker-v2-m3`. Its exact
contract is:

- original query text paired with exact chunk `display_text`;
- passage-only truncation at max pair length 1024;
- 20 candidates, raw model logit, batch size 4, CPU by default;
- evaluation depth 10 and serving depth 8;
- score descending, prior fused rank ascending, chunk ID ascending;
- non-finite scores rejected;
- query-level atomic checkpoints with model/config/corpus/selection/candidate
  fingerprints.

The adapter is lazy and Phase 8 DEV fusion does not load this model. The DEV
run used immutable revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` and
completed 160 query checkpoints. The fixed selection rule selected weighted
RRF followed by the reranker: nDCG@10 improved by 0.065177, Recall@10 by
0.056738, and CompleteEvidenceRecall@10 by 0.056738. Query latency was not
recorded by the completed run; token diagnostics and checkpoint reuse status
are preserved in the text-free DEV metrics artifact.

## Metadata filtering

Filters are explicit typed constraints over jurisdiction, issuing authority,
document type, inclusive publication date range, legal status, and regulation
name. Values OR within a field and fields AND across one another. Unknown
metadata never satisfies an explicit constraint. Filters are hard eligibility
masks before top-k selection and no natural-language filter extraction is
implemented.

The current Phase-7 retrieval release does not carry structured document
metadata, so Phase 8 reports null coverage rather than inferring fields from
text or using qrels. An independently supplied structured metadata index can
populate the same coverage contract later.

## Anti-overfitting and compute boundaries

Phase 8 may legitimately produce a negative result. If the fixed ladder fails,
the experiment does not automatically add weights, models, candidate depths,
or rerankers. DEV and holdout populations remain protected by the Phase-7
loader; this task runs no holdout. Existing E5/BGE corpus embeddings are
reused and never regenerated. The future one-shot holdout command is
`uv run kawaneen retrieval phase8-holdout --allow-holdout --resume --device cpu`.
It is permission-gated and writes private per-query rankings, scores,
provenance, relevance/evidence indicators, slices, robustness links, and
latency before aggregate holdout analysis. No full repository pytest/coverage
run or Phase 9 work belongs in this phase task.
