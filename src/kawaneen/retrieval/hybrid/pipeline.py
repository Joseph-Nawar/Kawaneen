"""Composition of frozen first-stage retrieval and optional reranking."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from kawaneen.retrieval.hybrid.contracts import (
    FusedCandidate,
    FusionConfig,
    RerankedCandidate,
    SourceHit,
)
from kawaneen.retrieval.hybrid.fusion import fuse_ranked_hits


def retrieve_and_fuse(
    query: str,
    *,
    sparse_search: Callable[[str, int], Sequence[SourceHit]],
    dense_search: Callable[[str, int], Sequence[SourceHit]],
    config: FusionConfig,
) -> tuple[FusedCandidate, ...]:
    return fuse_ranked_hits(
        sparse=sparse_search(query, config.sparse_top_k),
        dense=dense_search(query, config.dense_top_k),
        config=config,
    )


def rerank_for_serving(
    candidates: Sequence[FusedCandidate], scores: Mapping[str, float]
) -> tuple[RerankedCandidate, ...]:
    reranked = [
        RerankedCandidate(
            chunk_id=candidate.chunk_id,
            score=float(scores[candidate.chunk_id]),
            prior_fused_rank=candidate.fused_rank,
        )
        for candidate in candidates
    ]
    reranked.sort(key=lambda item: (-item.score, item.prior_fused_rank, item.chunk_id))
    return tuple(reranked[:8])
