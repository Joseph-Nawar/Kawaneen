# pyright: basic, reportArgumentType=false
"""Deterministic reciprocal-rank fusion."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from kawaneen.retrieval.hybrid.contracts import FusedCandidate, FusionConfig, SourceHit


def fuse_ranked_hits(
    *, sparse: Sequence[SourceHit], dense: Sequence[SourceHit], config: FusionConfig
) -> tuple[FusedCandidate, ...]:
    sparse_hits = tuple(sparse[: config.sparse_top_k])
    dense_hits = tuple(dense[: config.dense_top_k])
    combined: OrderedDict[str, dict[str, object]] = OrderedDict()
    for rank, hit in enumerate(sparse_hits, start=1):
        if hit.chunk_id in combined:
            raise ValueError(f"duplicate chunk ID in sparse ranking: {hit.chunk_id}")
        combined[hit.chunk_id] = {
            "sparse_rank": rank,
            "sparse_score": hit.score,
            "dense_rank": None,
            "dense_score": None,
        }
    for rank, hit in enumerate(dense_hits, start=1):
        record = combined.setdefault(
            hit.chunk_id,
            {
                "sparse_rank": None,
                "sparse_score": None,
                "dense_rank": None,
                "dense_score": None,
            },
        )
        if record["dense_rank"] is not None:
            raise ValueError(f"duplicate chunk ID in dense ranking: {hit.chunk_id}")
        record["dense_rank"] = rank
        record["dense_score"] = hit.score
    candidates: list[FusedCandidate] = []
    for chunk_id, record in combined.items():
        sparse_rank = record["sparse_rank"]
        dense_rank = record["dense_rank"]
        score = 0.0
        if sparse_rank is not None:
            score += config.sparse_weight / (config.rrf_k + int(sparse_rank))
        if dense_rank is not None:
            score += config.dense_weight / (config.rrf_k + int(dense_rank))
        provenance = (
            "both"
            if sparse_rank is not None and dense_rank is not None
            else "sparse-only"
            if sparse_rank is not None
            else "dense-only"
        )
        candidates.append(
            FusedCandidate(
                chunk_id=chunk_id,
                fused_rank=0,
                fused_score=score,
                sparse_rank=int(sparse_rank) if sparse_rank is not None else None,
                sparse_score=float(record["sparse_score"])
                if record["sparse_score"] is not None
                else None,
                dense_rank=int(dense_rank) if dense_rank is not None else None,
                dense_score=float(record["dense_score"])
                if record["dense_score"] is not None
                else None,
                provenance=provenance,
            )
        )
    candidates.sort(
        key=lambda item: (
            -item.fused_score,
            min(rank for rank in (item.sparse_rank, item.dense_rank) if rank is not None),
            item.chunk_id,
        )
    )
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
        for rank, item in enumerate(candidates[: config.candidate_k], start=1)
    )
