from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from kawaneen.demo.corpus import MODEL_REVISION, DemoCorpus
from kawaneen.retrieval.bm25 import BM25Index
from kawaneen.retrieval.dense_models import E5SmallAdapter
from kawaneen.retrieval.hybrid.contracts import FusedCandidate, SourceHit
from kawaneen.retrieval.hybrid.reranker import BGERerankerAdapter
from kawaneen.retrieval.serving import (
    ServingEvidence,
    ServingRetrievalResult,
    ServingRetrievalSummary,
)
from kawaneen.retrieval.vector_index import NumpyExactIndex

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def fuse_demo_hits(
    sparse: Sequence[SourceHit], dense: Sequence[SourceHit], *, candidate_k: int = 8
) -> tuple[FusedCandidate, ...]:
    combined: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(sparse[:12], 1):
        combined[hit.chunk_id] = {"sparse_rank": rank, "sparse_score": hit.score}
    for rank, hit in enumerate(dense[:12], 1):
        record = combined.setdefault(hit.chunk_id, {})
        record["dense_rank"] = rank
        record["dense_score"] = hit.score
    candidates: list[FusedCandidate] = []
    for chunk_id, record in combined.items():
        sparse_rank = record.get("sparse_rank")
        dense_rank = record.get("dense_rank")
        score = (1 / (60 + int(sparse_rank)) if sparse_rank is not None else 0.0) + (
            1 / (60 + int(dense_rank)) if dense_rank is not None else 0.0
        )
        candidates.append(
            FusedCandidate(
                chunk_id=chunk_id,
                fused_rank=0,
                fused_score=score,
                sparse_rank=int(sparse_rank) if sparse_rank is not None else None,
                sparse_score=float(record["sparse_score"]) if "sparse_score" in record else None,
                dense_rank=int(dense_rank) if dense_rank is not None else None,
                dense_score=float(record["dense_score"]) if "dense_score" in record else None,
                provenance=(
                    "both"
                    if sparse_rank is not None and dense_rank is not None
                    else "sparse-only"
                    if sparse_rank is not None
                    else "dense-only"
                ),
            )
        )
    candidates.sort(key=lambda item: (-item.fused_score, item.chunk_id))
    return tuple(
        FusedCandidate(
            chunk_id=item.chunk_id,
            fused_rank=rank,
            fused_score=item.fused_score,
            sparse_rank=item.sparse_rank,
            sparse_score=item.sparse_score,
            dense_rank=item.dense_rank,
            dense_score=item.dense_score,
            provenance=item.provenance,
        )
        for rank, item in enumerate(candidates[:candidate_k], 1)
    )


class DemoRetriever:
    def __init__(
        self,
        corpus: DemoCorpus,
        *,
        query_encoder: Callable[[str], np.ndarray] | None = None,
        reranker: BGERerankerAdapter | None = None,
    ) -> None:
        self.corpus = corpus
        self.sparse = BM25Index.build(corpus.chunks.values(), "arabic-raw-v1")
        self.dense = NumpyExactIndex.build(corpus.vectors, tuple(corpus.chunks))
        self.query_encoder = query_encoder
        self.dense_adapter = (
            None if query_encoder is not None else E5SmallAdapter(revision=MODEL_REVISION)
        )
        self.reranker = reranker
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        if self.dense_adapter is not None:
            self.dense_adapter.preload()
        preload = getattr(self.reranker, "preload", None)
        if callable(preload):
            preload()
        self._initialized = True

    def search(self, query: str, limit: int = 5) -> ServingRetrievalResult:
        limit = min(max(1, limit), 5)
        sparse = tuple(
            SourceHit(hit.chunk_id, hit.score)
            for hit in self.sparse.search(query, top_k=12)
            if hit.score > 0
        )
        if self.query_encoder is None:
            if self.dense_adapter is None:
                raise RuntimeError("demo dense adapter is not configured")
            vector = self.dense_adapter.encode_queries((query,), batch_size=1)[0]
        else:
            vector = np.asarray(self.query_encoder(query), dtype=np.float32)
        dense = tuple(
            SourceHit(hit.chunk_id, hit.score) for hit in self.dense.search(vector, top_k=12)
        )
        candidates = fuse_demo_hits(sparse, dense)
        ranked_candidates = candidates
        score_type = "rrf_score"
        reranker_depth = 0
        reranker_scores: dict[str, float] = {}
        if self.reranker is not None:
            reranker_depth = min(4, len(candidates))
            pairs = tuple(
                (query, self.corpus.chunks[item.chunk_id].display_text)
                for item in candidates[:reranker_depth]
            )
            reranker_scores = {
                item.chunk_id: score
                for item, score in zip(
                    candidates[:reranker_depth],
                    self.reranker.score_pairs(pairs, batch_size=1),
                    strict=True,
                )
            }
            head = tuple(
                sorted(
                    candidates[:reranker_depth],
                    key=lambda item: (
                        -reranker_scores[item.chunk_id],
                        item.fused_rank,
                        item.chunk_id,
                    ),
                )
            )
            ranked_candidates = head + candidates[reranker_depth:]
            score_type = "mixed" if reranker_depth < len(candidates) else "reranker_raw_logit"
        evidence = tuple(
            self._evidence(
                item,
                rank,
                score=reranker_scores.get(item.chunk_id, item.fused_score),
                score_type=(
                    "reranker_raw_logit" if item.chunk_id in reranker_scores else "rrf_score"
                ),
            )
            for rank, item in enumerate(ranked_candidates[:limit], 1)
        )
        return ServingRetrievalResult(
            evidence=evidence,
            summary=ServingRetrievalSummary(
                strategy="demo_retrieval_first",
                sparse_top_k=12,
                dense_top_k=12,
                fused_candidate_count=8,
                reranker_depth=reranker_depth,
                top_score=evidence[0].score if evidence else None,
                hit_count=len(candidates),
                returned_count=len(evidence),
                score_type=score_type,
            ),
        )

    def _evidence(
        self,
        candidate: FusedCandidate,
        rank: int,
        *,
        score: float,
        score_type: str,
    ) -> ServingEvidence:
        chunk = self.corpus.chunks[candidate.chunk_id]
        metadata = self.corpus.metadata[candidate.chunk_id]
        return ServingEvidence(
            chunk_id=chunk.chunk_id,
            rank=rank,
            text=chunk.display_text,
            document_id=chunk.document_id,
            document_title=metadata.get("document_title"),
            article=metadata.get("article"),
            page=metadata.get("page"),
            source_url=None,
            score=score,
            score_type=score_type,
            provenance=candidate.provenance,
        )


__all__ = ["DemoRetriever", "fuse_demo_hits"]
