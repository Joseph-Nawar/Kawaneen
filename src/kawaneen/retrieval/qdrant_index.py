# pyright: basic
"""Minimal exact-search Qdrant adapter for the full local deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kawaneen.retrieval.models import ScoredChunk
from kawaneen.retrieval.vector_index import validate_normalized_vectors


@dataclass(frozen=True, slots=True)
class _ExactSearchParams:
    exact: bool = True


def _search_params() -> object:
    try:
        from qdrant_client.http import models

        return models.SearchParams(exact=True)
    except ImportError:
        return _ExactSearchParams()


@dataclass(frozen=True, slots=True)
class QdrantExactIndex:
    """Qdrant-backed dense index that always requests exact search."""

    client: Any
    collection_name: str
    dimension: int
    chunk_ids: tuple[str, ...]

    @classmethod
    def build(
        cls,
        *,
        client: Any,
        collection_name: str,
        vectors: np.ndarray,
        chunk_ids: tuple[str, ...] | list[str],
    ) -> QdrantExactIndex:
        values = np.asarray(vectors, dtype=np.float32)
        validate_normalized_vectors(values)
        ids = tuple(chunk_ids)
        if values.shape[0] != len(ids) or len(ids) != len(set(ids)):
            raise ValueError("vector rows and unique chunk IDs must match")
        if not collection_name:
            raise ValueError("Qdrant collection name must not be empty")
        return cls(client, collection_name, int(values.shape[1]), ids)

    def search(self, query: np.ndarray, *, top_k: int = 10) -> tuple[ScoredChunk, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        value = np.asarray(query, dtype=np.float32)
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        validate_normalized_vectors(value.reshape(1, -1))
        if value.shape[0] != self.dimension:
            raise ValueError("query dimension does not match index")
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=value.tolist(),
            limit=top_k,
            search_params=_search_params(),
            with_payload=True,
        )
        points = getattr(response, "points", None)
        if not isinstance(points, (list, tuple)):
            raise ValueError("Qdrant response points are malformed")
        hits: list[ScoredChunk] = []
        for point in points:
            payload = getattr(point, "payload", None)
            chunk_id = payload.get("chunk_id") if isinstance(payload, dict) else None
            score = getattr(point, "score", None)
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Qdrant response payload has no chunk_id")
            if not isinstance(score, (int, float)) or not np.isfinite(score):
                raise ValueError("Qdrant response score is malformed")
            if chunk_id not in self.chunk_ids:
                raise ValueError("Qdrant returned an unknown chunk_id")
            hits.append(ScoredChunk(chunk_id=chunk_id, score=float(score)))
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id)))[:top_k]


__all__ = ["QdrantExactIndex"]
