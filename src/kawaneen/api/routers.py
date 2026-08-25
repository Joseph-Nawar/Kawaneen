# pyright: reportUnusedFunction=false, reportArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false
# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
"""HTTP routes; all domain work is delegated to injected serving services."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from kawaneen.api.contracts import (
    AnswerRequest,
    AnswerResponse,
    Citation,
    ComponentStatus,
    DocumentDetail,
    DocumentPage,
    DocumentSummary,
    DocumentUnit,
    ErrorResponse,
    ExtractionResponse,
    ExtractRequest,
    HealthResponse,
    ModelCapability,
    ModelsResponse,
    RetrievalSummary,
    SearchRequest,
    SearchResponse,
)
from kawaneen.api.dependencies import get_container
from kawaneen.api.errors import (
    document_not_found,
    model_unavailable,
    request_timeout,
    service_unavailable,
)
from kawaneen.api.runtime import ServiceContainer
from kawaneen.extraction.serving import ModelUnavailableError
from kawaneen.generation.serving import (
    GenerationModelUnavailableError,
    callable_retriever,
)

_CONTAINER = Depends(get_container)
_POST_ERRORS: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


def build_router(
    *, search_timeout: float, answer_timeout: float, extract_timeout: float
) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.post(
        "/search",
        response_model=SearchResponse,
        operation_id="search_v1",
        responses=_POST_ERRORS,
    )
    async def search(
        request: SearchRequest,
        raw_request: Request,
        container: ServiceContainer = _CONTAINER,
    ) -> SearchResponse:
        service = container.retriever
        if service is None:
            raise service_unavailable("retrieval service is not ready")
        started = time.perf_counter()
        result = await _run_with_timeout(
            lambda: callable_retriever(service)(request.query, request.limit), search_timeout
        )
        return SearchResponse(
            request_id=_request_id(raw_request),
            results=tuple(_evidence(item) for item in result.evidence),
            retrieval=_summary(result.summary),
            latency_ms=(time.perf_counter() - started) * 1000,
            warnings=result.warnings,
        )

    @router.post(
        "/answer",
        response_model=AnswerResponse,
        operation_id="answer_v1",
        responses=_POST_ERRORS,
    )
    async def answer(
        request: AnswerRequest,
        raw_request: Request,
        container: ServiceContainer = _CONTAINER,
    ) -> AnswerResponse:
        service = container.answerer
        if service is None:
            raise service_unavailable("answer service is not ready")
        started = time.perf_counter()
        try:
            result = await _run_with_timeout(
                lambda: _call_answer(service, request.query), answer_timeout
            )
        except GenerationModelUnavailableError as error:
            raise model_unavailable() from error
        return AnswerResponse(
            request_id=_request_id(raw_request),
            answerable=result.answerable,
            answer=result.answer,
            abstention_reason=result.abstention_reason,
            citations=tuple(_citation(item) for item in result.citations),
            retrieval=_summary(result.retrieval.summary),
            latency_ms=(time.perf_counter() - started) * 1000,
            warnings=result.warnings,
        )

    @router.post(
        "/extract",
        response_model=ExtractionResponse,
        operation_id="extract_v1",
        responses=_POST_ERRORS,
    )
    async def extract(
        request: ExtractRequest,
        raw_request: Request,
        container: ServiceContainer = _CONTAINER,
    ) -> ExtractionResponse:
        service = container.extractor
        if service is None:
            raise service_unavailable("extraction service is not ready")
        started = time.perf_counter()
        try:
            result = await _run_with_timeout(
                lambda: service.extract(request.text, mode=request.mode.value), extract_timeout
            )
        except ModelUnavailableError as error:
            raise model_unavailable() from error
        return ExtractionResponse(
            request_id=_request_id(raw_request),
            result=result.result,
            capability_status=result.capability_status,
            latency_ms=(time.perf_counter() - started) * 1000,
            warnings=result.warnings,
        )

    @router.get(
        "/documents",
        response_model=DocumentPage,
        operation_id="list_documents_v1",
        responses={503: {"model": ErrorResponse}},
    )
    async def documents(
        raw_request: Request,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=20, ge=1, le=100),
        container: ServiceContainer = _CONTAINER,
    ) -> DocumentPage:
        if container.corpus is None:
            raise service_unavailable("canonical corpus is not ready")
        page = await _run_with_timeout(
            lambda: container.corpus.list_documents(offset=offset, limit=limit),
            search_timeout,
        )
        return DocumentPage(
            request_id=_request_id(raw_request),
            items=tuple(_document_summary(item) for item in page.items),
            offset=page.offset,
            limit=page.limit,
            total=page.total,
        )

    @router.get(
        "/documents/{document_id}",
        response_model=DocumentDetail,
        operation_id="get_document_v1",
        responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def document(
        document_id: str,
        raw_request: Request,
        container: ServiceContainer = _CONTAINER,
    ) -> DocumentDetail:
        if "/" in document_id or "\\" in document_id or document_id in {".", ".."}:
            raise document_not_found(document_id)
        if container.corpus is None:
            raise service_unavailable("canonical corpus is not ready")
        value = await _run_with_timeout(
            lambda: container.corpus.get_document(document_id),
            search_timeout,
        )
        if value is None:
            raise document_not_found(document_id)
        return DocumentDetail(
            request_id=_request_id(raw_request),
            document=_document_summary(value),
            units=tuple(_document_unit(item) for item in value.units),
        )

    @router.get("/health", response_model=HealthResponse, operation_id="health_v1")
    async def health(
        raw_request: Request,
        container: ServiceContainer = _CONTAINER,
    ) -> HealthResponse | JSONResponse:
        snapshot = container.health()
        response = HealthResponse(
            request_id=_request_id(raw_request),
            status="ready" if snapshot.status == "ready" else "degraded",
            components=tuple(
                ComponentStatus(
                    name=item.name,
                    ready=item.ready,
                    required=item.required,
                    detail=item.detail,
                )
                for item in snapshot.components
            ),
        )
        if response.status == "degraded":
            return JSONResponse(status_code=503, content=response.model_dump(mode="json"))
        return response

    @router.get("/models", response_model=ModelsResponse, operation_id="models_v1")
    async def models(
        raw_request: Request,
        container: ServiceContainer = _CONTAINER,
    ) -> ModelsResponse:
        return ModelsResponse(
            request_id=_request_id(raw_request),
            capabilities=tuple(
                ModelCapability(
                    capability=item.capability,
                    provider=item.provider,
                    model=item.model,
                    revision=item.revision,
                    loaded=item.loaded,
                    ready=item.ready,
                )
                for item in container.models()
            ),
        )

    return router


async def _run_with_timeout(function: Callable[[], Any], timeout: float) -> Any:
    try:
        return await asyncio.wait_for(asyncio.to_thread(function), timeout=timeout)
    except TimeoutError as error:
        raise request_timeout() from error


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "missing-request-id"))


def _call_answer(service: object, query: str) -> Any:
    method = getattr(service, "answer", None)
    return method(query) if callable(method) else service(query)  # type: ignore[operator]


def _evidence(item: Any) -> Any:
    from kawaneen.api.contracts import Evidence

    return Evidence(
        chunk_id=item.chunk_id,
        rank=item.rank,
        text=item.text,
        document_id=item.document_id,
        document_title=item.document_title,
        article=item.article,
        page=item.page,
        source_url=item.source_url,
        score=item.score,
        provenance=item.provenance,
    )


def _summary(item: Any) -> RetrievalSummary:
    return RetrievalSummary(
        sparse_top_k=item.sparse_top_k,
        dense_top_k=item.dense_top_k,
        fused_candidate_count=item.fused_candidate_count,
        reranker_depth=item.reranker_depth,
        returned_count=item.returned_count,
    )


def _citation(item: Any) -> Citation:
    return Citation(
        evidence_id=item.evidence_id,
        document_id=item.document_id,
        document_title=item.document_title,
        article=item.article,
        page=item.page,
        source_url=item.source_url,
        quoted_text=item.quoted_text,
    )


def _document_summary(item: Any) -> DocumentSummary:
    return DocumentSummary(
        document_id=item.document_id,
        title=item.title,
        source_id=item.source_id,
        jurisdiction=item.jurisdiction,
        unit_count=len(item.units),
    )


def _document_unit(item: Any) -> DocumentUnit:
    return DocumentUnit(
        unit_id=item.unit_id,
        ordinal=item.ordinal,
        unit_type=item.unit_type,
        text=item.text,
        heading_path=item.heading_path,
    )
