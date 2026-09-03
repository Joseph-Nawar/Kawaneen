"""Strict, public HTTP contracts for the versioned Kawaneen API."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kawaneen.extraction.contracts import ExtractionResult


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Jurisdiction(StrEnum):
    SA = "SA"


class ExtractionMode(StrEnum):
    DETERMINISTIC = "deterministic"
    HYBRID = "hybrid"


class SearchRequest(ApiModel):
    query: str = Field(min_length=1, max_length=2_000, examples=["ما هي مدة الاعتراض؟"])
    jurisdiction: Literal["SA"] = Field(default="SA", examples=["SA"])
    limit: int = Field(default=8, ge=1, le=8, examples=[8])

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("query must not be blank")
        return value


class AnswerRequest(ApiModel):
    query: str = Field(min_length=1, max_length=2_000, examples=["ما هي مدة الاعتراض؟"])
    jurisdiction: Literal["SA"] = "SA"

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("query must not be blank")
        return value


class ExtractRequest(ApiModel):
    text: str = Field(
        min_length=1, max_length=20_000, examples=["يلتزم الطرف بالسداد خلال ثلاثين يوماً."]
    )
    jurisdiction: Literal["SA"] = "SA"
    mode: ExtractionMode = ExtractionMode.HYBRID

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("text must not be blank")
        return value


class Evidence(ApiModel):
    chunk_id: str = Field(min_length=1)
    rank: int = Field(ge=1, le=8)
    text: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_title: str | None = None
    article: str | None = None
    page: str | None = None
    source_url: str | None = None
    score: float
    score_type: Literal["reranker_raw_logit", "rrf_score"] = "reranker_raw_logit"
    provenance: Literal["sparse-only", "dense-only", "both"] | None = None


class RetrievalSummary(ApiModel):
    strategy: Literal["hybrid_reranked", "demo_retrieval_first"] = "hybrid_reranked"
    sparse_top_k: int = 50
    dense_top_k: int = 50
    fused_candidate_count: int = 20
    reranker_depth: int = 8
    top_score: float | None = None
    hit_count: int = Field(ge=0, le=20)
    returned_count: int = Field(ge=0, le=8)
    score_type: Literal["reranker_raw_logit", "rrf_score", "mixed"] = "reranker_raw_logit"


class SearchResponse(ApiModel):
    request_id: str = Field(min_length=1, max_length=128)
    jurisdiction: Literal["SA"] = "SA"
    results: tuple[Evidence, ...]
    retrieval: RetrievalSummary
    latency_ms: float = Field(ge=0)
    warnings: tuple[str, ...] = ()


class Citation(ApiModel):
    evidence_id: str = Field(pattern=r"^E[0-9]{3,}$")
    document_id: str = Field(min_length=1)
    document_title: str | None = None
    article: str | None = None
    page: str | None = None
    source_url: str | None = None
    quoted_text: str = Field(min_length=1)


class AnswerResponse(ApiModel):
    request_id: str = Field(min_length=1, max_length=128)
    jurisdiction: Literal["SA"] = "SA"
    answerable: bool
    answer: str | None = None
    abstention_reason: str | None = None
    citations: tuple[Citation, ...] = ()
    retrieval: RetrievalSummary
    latency_ms: float = Field(ge=0)
    warnings: tuple[str, ...] = ()


class ExtractionResponse(ApiModel):
    request_id: str = Field(min_length=1, max_length=128)
    result: ExtractionResult
    capability_status: Literal["operational_candidates", "experimental_limited"]
    latency_ms: float = Field(ge=0)
    warnings: tuple[str, ...] = ()


class DocumentUnit(ApiModel):
    unit_id: str = Field(min_length=1)
    ordinal: int | None = Field(default=None, ge=1)
    unit_type: str = Field(min_length=1)
    text: str = Field(min_length=1)
    heading_path: tuple[str, ...] = ()


class DocumentSummary(ApiModel):
    document_id: str = Field(min_length=1)
    title: str = ""
    source_id: str | None = None
    jurisdiction: str | None = None
    unit_count: int = Field(ge=0)


class DocumentPage(ApiModel):
    request_id: str = Field(min_length=1, max_length=128)
    items: tuple[DocumentSummary, ...]
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    total: int = Field(ge=0)


class DocumentDetail(ApiModel):
    request_id: str = Field(min_length=1, max_length=128)
    document: DocumentSummary
    units: tuple[DocumentUnit, ...]


class ComponentStatus(ApiModel):
    name: str = Field(min_length=1)
    ready: bool
    required: bool
    detail: str | None = None


class HealthResponse(ApiModel):
    request_id: str = Field(min_length=1, max_length=128)
    status: Literal["ready", "degraded"]
    components: tuple[ComponentStatus, ...]


class ModelCapability(ApiModel):
    capability: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str | None = None
    revision: str | None = None
    loaded: bool
    ready: bool


class ModelsResponse(ApiModel):
    request_id: str = Field(min_length=1, max_length=128)
    capabilities: tuple[ModelCapability, ...]


ErrorCode = Literal[
    "VALIDATION_ERROR",
    "REQUEST_TOO_LARGE",
    "DOCUMENT_NOT_FOUND",
    "SERVICE_UNAVAILABLE",
    "MODEL_UNAVAILABLE",
    "REQUEST_TIMEOUT",
    "RATE_LIMITED",
    "INTERNAL_ERROR",
]


class ErrorDetail(ApiModel):
    code: ErrorCode
    message: str = Field(min_length=1, max_length=500)


class ErrorResponse(ApiModel):
    error: ErrorDetail
    request_id: str = Field(min_length=1, max_length=128)
