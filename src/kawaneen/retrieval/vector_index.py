# pyright: basic
"""Exact cosine/IP vector indexes with NumPy and optional Faiss backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from kawaneen.retrieval.models import ScoredChunk


def validate_normalized_vectors(vectors: np.ndarray) -> None:
    values = np.asarray(vectors)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("vectors must be a two-dimensional non-empty matrix")
    if values.dtype != np.float32:
        raise ValueError("vectors must use float32 dtype")
    if not np.all(np.isfinite(values)):
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("vectors must be L2 normalized")


def _rank(ids: Sequence[str], scores: np.ndarray, indices: np.ndarray) -> tuple[ScoredChunk, ...]:
    hits = [
        ScoredChunk(chunk_id=ids[int(index)], score=float(score))
        for index, score in zip(indices, scores, strict=True)
    ]
    return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id)))


@dataclass(frozen=True, slots=True)
class NumpyExactIndex:
    vectors: np.ndarray
    chunk_ids: tuple[str, ...]

    @classmethod
    def build(cls, vectors: np.ndarray, chunk_ids: Sequence[str]) -> NumpyExactIndex:
        values = np.asarray(vectors, dtype=np.float32)
        validate_normalized_vectors(values)
        ids = tuple(chunk_ids)
        if values.shape[0] != len(ids) or len(ids) != len(set(ids)):
            raise ValueError("vector rows and unique chunk IDs must match")
        return cls(vectors=values.copy(), chunk_ids=ids)

    def search(self, query: np.ndarray, *, top_k: int = 10) -> tuple[ScoredChunk, ...]:
        value = np.asarray(query, dtype=np.float32)
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        validate_normalized_vectors(value.reshape(1, -1))
        if value.shape[0] != self.vectors.shape[1]:
            raise ValueError("query dimension does not match index")
        scores = self.vectors @ value
        indices = np.arange(len(self.chunk_ids))
        return _rank(self.chunk_ids, scores, indices)[:top_k]


@dataclass(frozen=True, slots=True)
class FaissExactIndex:
    chunk_ids: tuple[str, ...]
    _index: Any

    @classmethod
    def build(cls, vectors: np.ndarray, chunk_ids: Sequence[str]) -> FaissExactIndex:
        values = np.asarray(vectors, dtype=np.float32)
        validate_normalized_vectors(values)
        ids = tuple(chunk_ids)
        if values.shape[0] != len(ids) or len(ids) != len(set(ids)):
            raise ValueError("vector rows and unique chunk IDs must match")
        try:
            import faiss  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required for the Faiss backend") from exc
        index = faiss.IndexFlatIP(values.shape[1])
        index.add(values)
        return cls(chunk_ids=ids, _index=index)

    def search(self, query: np.ndarray, *, top_k: int = 10) -> tuple[ScoredChunk, ...]:
        value = np.asarray(query, dtype=np.float32).reshape(1, -1)
        validate_normalized_vectors(value)
        scores, indices = self._index.search(value, len(self.chunk_ids))
        return _rank(self.chunk_ids, scores[0], indices[0])[:top_k]
