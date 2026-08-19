"""Typed contracts for Phase 8 candidate fusion and reranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Provenance = Literal["sparse-only", "dense-only", "both"]


@dataclass(frozen=True, slots=True)
class SourceHit:
    chunk_id: str
    score: float

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if not math.isfinite(self.score):
            raise ValueError("source scores must be finite")


@dataclass(frozen=True, slots=True)
class FusionConfig:
    sparse_weight: float = 1.0
    dense_weight: float = 1.0
    rrf_k: int = 60
    sparse_top_k: int = 50
    dense_top_k: int = 50
    candidate_k: int = 20

    def __post_init__(self) -> None:
        if self.sparse_weight < 0 or self.dense_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if not math.isfinite(self.sparse_weight) or not math.isfinite(self.dense_weight):
            raise ValueError("fusion weights must be finite")
        if self.rrf_k != 60:
            raise ValueError("Phase 8 fixes rrf_k at 60")
        if self.sparse_top_k != 50 or self.dense_top_k != 50:
            raise ValueError("Phase 8 fixes sparse and dense top-k at 50")
        if self.candidate_k != 20:
            raise ValueError("Phase 8 fixes fused candidates at 20")


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    chunk_id: str
    fused_rank: int
    fused_score: float
    sparse_rank: int | None
    sparse_score: float | None
    dense_rank: int | None
    dense_score: float | None
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    model_id: str = "BAAI/bge-reranker-v2-m3"
    model_revision: str = ""
    max_length: int = 1024
    candidate_count: int = 20
    batch_size: int = 4
    device: str = "cpu"
    evaluation_depth: int = 10
    serving_depth: int = 8
    scoring_contract: str = "raw-logit-v1"

    def __post_init__(self) -> None:
        if self.max_length < 1 or self.candidate_count != 20:
            raise ValueError("Phase 8 fixes reranker max length and candidate count")
        if self.batch_size < 1 or self.evaluation_depth != 10 or self.serving_depth != 8:
            raise ValueError("Phase 8 reranker depth and batch contracts are fixed")


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    chunk_id: str
    score: float
    prior_fused_rank: int
