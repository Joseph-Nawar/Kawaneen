"""Deterministic Jaccard keyword retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from kawaneen.retrieval.models import RetrievalChunk, ScoredChunk
from kawaneen.retrieval.tokenization import represent


@dataclass(frozen=True, slots=True)
class KeywordIndex:
    policy_id: str
    chunks: tuple[RetrievalChunk, ...]
    token_sets: tuple[frozenset[str], ...]

    @classmethod
    def build(cls, chunks: Iterable[RetrievalChunk], policy_id: str) -> KeywordIndex:
        selected = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        ids = [chunk.chunk_id for chunk in selected]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk IDs are not allowed")
        token_sets = tuple(
            frozenset(represent(chunk.display_text, policy_id).tokens) for chunk in selected
        )
        return cls(policy_id=policy_id, chunks=selected, token_sets=token_sets)

    def score_query(self, query: str) -> tuple[ScoredChunk, ...]:
        query_tokens = frozenset(represent(query, self.policy_id).tokens)
        scores: list[ScoredChunk] = []
        for chunk, document_tokens in zip(self.chunks, self.token_sets, strict=True):
            union = query_tokens | document_tokens
            score = len(query_tokens & document_tokens) / len(union) if union else 0.0
            scores.append(ScoredChunk(chunk_id=chunk.chunk_id, score=score))
        return tuple(sorted(scores, key=lambda hit: (-hit.score, hit.chunk_id)))

    def search(self, query: str, *, top_k: int = 10) -> tuple[ScoredChunk, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        return self.score_query(query)[:top_k]
