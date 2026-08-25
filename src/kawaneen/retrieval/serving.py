"""Serving-only retrieval composition over frozen Phase 7/8 primitives."""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from kawaneen.retrieval.hybrid.contracts import FusedCandidate, FusionConfig, SourceHit
from kawaneen.retrieval.hybrid.pipeline import rerank_for_serving, retrieve_and_fuse
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
    strategy: str = "hybrid_reranked"
    sparse_top_k: int = 50
    dense_top_k: int = 50
    fused_candidate_count: int = 20
    reranker_depth: int = 8
    top_score: float | None = None
    hit_count: int = 0
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
        fusion_config: FusionConfig,
        metadata: Mapping[str, Mapping[str, str | None]] | None = None,
    ) -> None:
        self.chunks = chunks
        self.sparse_search = sparse_search
        self.dense_search = dense_search
        self.reranker = reranker
        self.fusion_config = fusion_config
        self.metadata = metadata or {}

    def search(self, query: str, limit: int = 8) -> ServingRetrievalResult:
        if not 1 <= limit <= 8:
            raise ValueError("serving retrieval limit must be between 1 and 8")
        candidates = retrieve_and_fuse(
            query,
            sparse_search=self.sparse_search,
            dense_search=self.dense_search,
            config=self.fusion_config,
        )
        scores = self.reranker(query, candidates)
        if any(candidate.chunk_id not in scores for candidate in candidates):
            raise ValueError("reranker did not score every fused candidate")
        ranked = rerank_for_serving(candidates, scores)
        candidate_by_id = {candidate.chunk_id: candidate for candidate in candidates}
        evidence = tuple(
            self._evidence(candidate_by_id[item.chunk_id], item.score, rank)
            for rank, item in enumerate(ranked[:limit], start=1)
        )
        return ServingRetrievalResult(
            evidence=evidence,
            summary=ServingRetrievalSummary(
                strategy="hybrid_reranked",
                top_score=ranked[0].score if ranked else None,
                hit_count=len(candidates),
                returned_count=len(evidence),
            ),
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


def load_serving_chunks(path: Path) -> dict[str, RetrievalChunk]:
    """Load retrieval chunks directly, without evaluation-release helpers."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise FileNotFoundError("serving retrieval corpus is unavailable") from error
    chunks: dict[str, RetrievalChunk] = {}
    for line in lines:
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("serving retrieval chunk must be an object")
        value = cast(dict[str, object], raw)
        chunk_id = _string(value.get("chunk_id"), "chunk_id")
        if not chunk_id or chunk_id in chunks:
            raise ValueError("serving retrieval chunks must have unique IDs")
        raw_unit_ids = value.get("source_unit_ids", ())
        if not isinstance(raw_unit_ids, list):
            raise ValueError(f"serving retrieval chunk source units are invalid: {chunk_id}")
        source_unit_ids = tuple(str(item) for item in cast(list[object], raw_unit_ids))
        if not source_unit_ids:
            raise ValueError(f"serving retrieval chunk has no source units: {chunk_id}")
        spans: list[tuple[int, int]] = []
        raw_spans = value.get("source_spans", ())
        if not isinstance(raw_spans, list):
            raise ValueError(f"serving retrieval chunk spans are invalid: {chunk_id}")
        for raw_span_value in raw_spans:
            if not isinstance(raw_span_value, dict):
                raise ValueError(f"serving retrieval chunk span is invalid: {chunk_id}")
            raw_span = cast(dict[str, object], raw_span_value)
            start = raw_span.get("start")
            end = raw_span.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                raise ValueError(f"serving retrieval chunk span is invalid: {chunk_id}")
            spans.append((start, end))
        chunks[chunk_id] = RetrievalChunk(
            chunk_id=chunk_id,
            document_id=_string(value.get("document_id"), "document_id"),
            source_id=_string(value.get("source_id"), "source_id"),
            unit_type=_string(value.get("unit_type"), "unit_type"),
            display_text=_string(value.get("display_text"), "display_text"),
            search_text=_string(value.get("search_text"), "search_text"),
            source_unit_ids=source_unit_ids,
            chunk_policy_hash=_string(value.get("chunk_policy_hash"), "chunk_policy_hash"),
            normalization_policy_id=_string(
                value.get("normalization_policy_id"), "normalization_policy_id"
            ),
            normalization_policy_hash=_string(
                value.get("normalization_policy_hash"), "normalization_policy_hash"
            ),
            token_count=_integer(value.get("token_count"), "token_count"),
            source_spans=tuple(spans),
        )
    if not chunks:
        raise ValueError("serving retrieval corpus is empty")
    return chunks


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"serving retrieval chunk {label} is invalid")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"serving retrieval chunk {label} is invalid")
    return value


__all__ = [
    "HybridServingRetriever",
    "ServingEvidence",
    "ServingRetrievalResult",
    "ServingRetrievalSummary",
    "load_serving_chunks",
]
