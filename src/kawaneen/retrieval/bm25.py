# pyright: basic
"""Okapi BM25 retrieval through bm25s with a deterministic reference fallback."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from kawaneen.retrieval.models import RetrievalChunk, ScoredChunk
from kawaneen.retrieval.tokenization import represent


@dataclass(frozen=True, slots=True)
class BM25Index:
    policy_id: str
    k1: float
    b: float
    chunks: tuple[RetrievalChunk, ...]
    token_lists: tuple[tuple[str, ...], ...]
    _backend: Any | None = None

    @classmethod
    def build(
        cls,
        chunks: Iterable[RetrievalChunk],
        policy_id: str,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> BM25Index:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters are outside the fixed Okapi range")
        selected = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        ids = [chunk.chunk_id for chunk in selected]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk IDs are not allowed")
        token_lists = tuple(
            tuple(represent(chunk.display_text, policy_id).tokens) for chunk in selected
        )
        backend: Any | None = None
        try:
            import bm25s

            backend = bm25s.BM25(k1=k1, b=b, method="lucene")
            backend.index(list(token_lists), show_progress=False)
        except ImportError:
            backend = None
        return cls(
            policy_id=policy_id,
            k1=k1,
            b=b,
            chunks=selected,
            token_lists=token_lists,
            _backend=backend,
        )

    def _reference_scores(self, query_tokens: tuple[str, ...]) -> tuple[ScoredChunk, ...]:
        document_count = len(self.token_lists)
        lengths = [len(tokens) for tokens in self.token_lists]
        average_length = sum(lengths) / max(document_count, 1)
        query_counts = Counter(query_tokens)
        document_frequencies = Counter(
            token for tokens in self.token_lists for token in set(tokens)
        )
        scores: list[ScoredChunk] = []
        for chunk, tokens, length in zip(self.chunks, self.token_lists, lengths, strict=True):
            term_counts = Counter(tokens)
            score = 0.0
            for token, query_frequency in query_counts.items():
                term_frequency = term_counts.get(token, 0)
                if not term_frequency:
                    continue
                df = document_frequencies[token]
                idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(average_length, 1.0)
                )
                score += idf * (term_frequency * (self.k1 + 1.0) / denominator) * query_frequency
            scores.append(ScoredChunk(chunk_id=chunk.chunk_id, score=score))
        return tuple(sorted(scores, key=lambda hit: (-hit.score, hit.chunk_id)))

    def score_query(self, query: str) -> tuple[ScoredChunk, ...]:
        query_tokens = represent(query, self.policy_id).tokens
        if self._backend is None:
            return self._reference_scores(query_tokens)
        result = self._backend.retrieve(
            [list(query_tokens)],
            k=len(self.chunks),
            show_progress=False,
            n_threads=1,
        )
        documents = result.documents[0]
        values = result.scores[0]
        hits = [
            ScoredChunk(chunk_id=self.chunks[int(index)].chunk_id, score=float(score))
            for index, score in zip(documents, values, strict=True)
        ]
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id)))

    def search(self, query: str, *, top_k: int = 10) -> tuple[ScoredChunk, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        return self.score_query(query)[:top_k]
