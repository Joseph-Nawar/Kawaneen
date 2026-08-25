"""Serving-only retrieval composition over frozen Phase 7/8 primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from kawaneen.retrieval.hybrid.contracts import FusedCandidate, SourceHit
from kawaneen.retrieval.hybrid.pipeline import retrieve_and_fuse
from kawaneen.retrieval.models import RetrievalChunk


@dataclass(frozen=True, slots=True)
class ServingEvidence:
    chunk_id: str
    rank: int
    text: str
    document_id: str
    document_title: str | None
    article: str | None
    page: str | None
    source_url: str | None
    score: float
    score_type: str = "reranker_raw_logit"
    provenance: str | None = None


@dataclass(frozen=True, slots=True)
class ServingRetrievalSummary:
    sparse_top_k: int = 50
    dense_top_k: int = 50
    fused_candidate_count: int = 20
    reranker_depth: int = 8
    returned_count: int = 0
    score_type: str = "reranker_raw_logit"


@dataclass(frozen=True, slots=True)
class ServingRetrievalResult:
    evidence: tuple[ServingEvidence, ...]
    summary: ServingRetrievalSummary
    warnings: tuple[str, ...] = ()


class Reranker(Protocol):
    def __call__(self, query: str, candidates: Sequence[FusedCandidate]) -> Mapping[str, float]: ...


class HybridServingRetriever:
    """Run the fixed sparse+dense+RRF+reranker pipeline for one query."""

    def __init__(
        self,
        *,
        chunks: Mapping[str, RetrievalChunk],
        sparse_search: Callable[[str, int], Sequence[SourceHit]],
        dense_search: Callable[[str, int], Sequence[SourceHit]],
        reranker: Reranker,
        metadata: Mapping[str, Mapping[str, str | None]] | None = None,
    ) -> None:
        self.chunks = chunks
        self.sparse_search = sparse_search
        self.dense_search = dense_search
        self.reranker = reranker
        self.metadata = metadata or {}

    def search(self, query: str, limit: int = 8) -> ServingRetrievalResult:
        if not 1 <= limit <= 8:
            raise ValueError("serving retrieval limit must be between 1 and 8")
        candidates = retrieve_and_fuse(
            query,
            sparse_search=self.sparse_search,
            dense_search=self.dense_search,
            config=_frozen_fusion_config(),
        )
        scores = self.reranker(query, candidates)
        if any(candidate.chunk_id not in scores for candidate in candidates):
            raise ValueError("reranker did not score every fused candidate")
        ranked = sorted(
            ((candidate, float(scores[candidate.chunk_id])) for candidate in candidates),
            key=lambda pair: (-pair[1], pair[0].fused_rank, pair[0].chunk_id),
        )[:8]
        evidence = tuple(
            self._evidence(candidate, score, rank)
            for rank, (candidate, score) in enumerate(ranked[:limit], start=1)
        )
        return ServingRetrievalResult(
            evidence=evidence,
            summary=ServingRetrievalSummary(returned_count=len(evidence)),
        )

    def _evidence(self, candidate: FusedCandidate, score: float, rank: int) -> ServingEvidence:
        chunk = self.chunks.get(candidate.chunk_id)
        if chunk is None:
            raise ValueError(
                f"retrieval chunk is missing from serving corpus: {candidate.chunk_id}"
            )
        metadata = self.metadata.get(candidate.chunk_id, {})
        return ServingEvidence(
            chunk_id=chunk.chunk_id,
            rank=rank,
            text=chunk.display_text,
            document_id=chunk.document_id,
            document_title=metadata.get("document_title"),
            article=metadata.get("article"),
            page=metadata.get("page"),
            source_url=metadata.get("source_url"),
            score=score,
            provenance=candidate.provenance,
        )


def _frozen_fusion_config():
    from kawaneen.retrieval.hybrid.contracts import FusionConfig

    return FusionConfig(
        sparse_top_k=50,
        dense_top_k=50,
        candidate_k=20,
        rrf_k=60,
    )


__all__ = [
    "HybridServingRetriever",
    "ServingEvidence",
    "ServingRetrievalResult",
    "ServingRetrievalSummary",
]
