"""Typed, safe HTTP boundary for the Phase 12 API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from kawaneen.api.contracts import (
    AnswerResponse,
    DocumentDetail,
    DocumentPage,
    ExtractionResponse,
    HealthResponse,
    ModelsResponse,
    SearchResponse,
)

_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


class UiClient(Protocol):
    def search(self, query: str, limit: int = 8) -> SearchResponse: ...

    def answer(self, query: str) -> AnswerResponse: ...

    def extract(self, text: str, mode: str = "deterministic") -> ExtractionResponse: ...

    def list_documents(self, offset: int = 0, limit: int = 20) -> DocumentPage: ...

    def get_document(self, document_id: str) -> DocumentDetail: ...

    def health(self) -> HealthResponse: ...

    def models(self) -> ModelsResponse: ...


@dataclass(frozen=True)
class UiApiError(Exception):
    code: str
    message: str
    status_code: int | None = None
    request_id: str | None = None

    def __str__(self) -> str:
        suffix = f" (request {self.request_id})" if self.request_id else ""
        return f"{self.code}: {self.message}{suffix}"


class HttpUiClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 65.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def search(self, query: str, limit: int = 8) -> SearchResponse:
        return self._post("/v1/search", {"query": query, "jurisdiction": "SA", "limit": limit}, SearchResponse)

    def answer(self, query: str) -> AnswerResponse:
        return self._post("/v1/answer", {"query": query, "jurisdiction": "SA"}, AnswerResponse)

    def extract(self, text: str, mode: str = "deterministic") -> ExtractionResponse:
        return self._post(
            "/v1/extract", {"text": text, "jurisdiction": "SA", "mode": mode}, ExtractionResponse
        )

    def list_documents(self, offset: int = 0, limit: int = 20) -> DocumentPage:
        return self._get("/v1/documents", {"offset": offset, "limit": limit}, DocumentPage)

    def get_document(self, document_id: str) -> DocumentDetail:
        return self._get(f"/v1/documents/{document_id}", None, DocumentDetail)

    def health(self) -> HealthResponse:
        return self._get("/v1/health", None, HealthResponse, accept_degraded=True)

    def models(self) -> ModelsResponse:
        return self._get("/v1/models", None, ModelsResponse)

    def _post(self, path: str, payload: Mapping[str, Any], model: type[_ResponseT]) -> _ResponseT:
        return self._request("POST", path, model, json=dict(payload))

    def _get(
        self,
        path: str,
        params: Mapping[str, Any] | None,
        model: type[_ResponseT],
        *,
        accept_degraded: bool = False,
    ) -> _ResponseT:
        return self._request("GET", path, model, params=params, accept_degraded=accept_degraded)

    def _request(
        self,
        method: str,
        path: str,
        model: type[_ResponseT],
        *,
        json: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        accept_degraded: bool = False,
    ) -> _ResponseT:
        request_id = str(uuid4())
        try:
            response = self._client.request(
                method,
                path,
                json=json,
                params=params,
                headers={"X-Request-ID": request_id},
            )
        except (httpx.TimeoutException, httpx.TransportError) as error:
            raise UiApiError("API_UNAVAILABLE", "The Phase 12 API could not be reached.") from error
        if response.status_code >= 400 and not (accept_degraded and response.status_code == 503):
            raise _api_error(response)
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            raise UiApiError("API_INVALID_RESPONSE", "The Phase 12 API returned an invalid response.") from error


def _api_error(response: httpx.Response) -> UiApiError:
    try:
        payload = response.json()
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        code = str(error.get("code", "API_ERROR"))
        message = str(error.get("message", "The Phase 12 API returned an error."))
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
    except ValueError:
        code = "API_ERROR"
        message = "The Phase 12 API returned an error."
        request_id = None
    return UiApiError(code, message[:500], response.status_code, request_id)
