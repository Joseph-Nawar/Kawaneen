# pyright: basic, reportAttributeAccessIssue=false, reportArgumentType=false
"""Hard eligibility masks applied before top-k selection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from kawaneen.retrieval.bm25 import BM25Index
from kawaneen.retrieval.models import ScoredChunk


def filtered_bm25(
    index: BM25Index, query: str, eligible_document_ids: set[str], *, top_k: int
) -> tuple[ScoredChunk, ...]:
    eligible = {
        chunk.chunk_id for chunk in index.chunks if chunk.document_id in eligible_document_ids
    }
    return tuple(hit for hit in index.score_query(query) if hit.chunk_id in eligible)[:top_k]


def filtered_dense(
    index: object, query: np.ndarray, eligible_chunk_ids: set[str], *, top_k: int
) -> tuple[ScoredChunk, ...]:
    if hasattr(index, "vectors") and hasattr(index, "chunk_ids"):
        ids = tuple(index.chunk_ids)
        selected = [
            position for position, chunk_id in enumerate(ids) if chunk_id in eligible_chunk_ids
        ]
        if not selected:
            return ()
        vectors = np.asarray(index.vectors)[selected]
        value = np.asarray(query, dtype=np.float32).reshape(-1)
        scores = vectors @ value
        hits = (
            ScoredChunk(ids[position], float(score))
            for position, score in zip(selected, scores, strict=True)
        )
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id)))[:top_k]
    hits = index.search(query, top_k=len(index.chunk_ids))
    return tuple(hit for hit in hits if hit.chunk_id in eligible_chunk_ids)[:top_k]


def document_ids_for_chunks(chunks: Iterable[object], eligible_document_ids: set[str]) -> set[str]:
    return {
        str(chunk.chunk_id) for chunk in chunks if str(chunk.document_id) in eligible_document_ids
    }
