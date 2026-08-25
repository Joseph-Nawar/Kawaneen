from __future__ import annotations

import time

from fastapi.testclient import TestClient

from kawaneen.api.runtime import ComponentReadiness, ServiceContainer
from kawaneen.corpus.serving import InMemoryCorpusRepository, ServingDocument, ServingUnit
from kawaneen.retrieval.serving import (
    ServingEvidence,
    ServingRetrievalResult,
    ServingRetrievalSummary,
)


def _retrieval() -> ServingRetrievalResult:
    return ServingRetrievalResult(
        evidence=(
            ServingEvidence(
                chunk_id="chunk-1",
                rank=1,
                text="The deadline is thirty days.",
                document_id="doc-1",
                document_title="Regulation",
                article="1",
                page=None,
                source_url=None,
                score=2.5,
            ),
        ),
        summary=ServingRetrievalSummary(returned_count=1),
    )


class FakeRetriever:
    def __init__(self, fail: bool = False, delay: float = 0) -> None:
        self.fail = fail
        self.delay = delay

    def search(self, query: str, limit: int) -> ServingRetrievalResult:
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise ValueError("local path must not escape")
        return _retrieval()


class FakeAnswerer:
    def answer(self, query: str):
        from kawaneen.generation.serving import ServingAnswerResult

        return ServingAnswerResult(
            answerable=True,
            answer="The deadline is thirty days.",
            abstention_reason=None,
            citations=(),
            retrieval=_retrieval(),
        )


def _container(*, retriever: object | None = None, delay: float = 0) -> ServiceContainer:
    from kawaneen.extraction.serving import ServingExtractor

    return ServiceContainer(
        retriever=retriever or FakeRetriever(delay=delay),
        answerer=FakeAnswerer(),
        extractor=ServingExtractor(provider=None),
        corpus=InMemoryCorpusRepository(
            (
                ServingDocument(
                    "doc-1", "Regulation", "source-1", (ServingUnit("u-1", "article", "text"),)
                ),
            )
        ),
        components=(ComponentReadiness("api", True, True),),
    )


def test_all_v1_endpoints_and_request_id_propagation() -> None:
    from kawaneen.api.app import create_app

    with TestClient(create_app(lambda: _container())) as client:
        request_id = "client:req-1"
        search = client.post(
            "/v1/search",
            headers={"X-Request-ID": request_id},
            json={"query": "deadline", "jurisdiction": "SA", "limit": 1},
        )
        answer = client.post("/v1/answer", json={"query": "deadline", "jurisdiction": "SA"})
        extraction = client.post(
            "/v1/extract",
            json={"text": "يلتزم الطرف بالسداد.", "jurisdiction": "SA", "mode": "deterministic"},
        )
        documents = client.get("/v1/documents")
        detail = client.get("/v1/documents/doc-1")
        health = client.get("/v1/health")
        models = client.get("/v1/models")

    assert search.status_code == 200
    assert search.headers["X-Request-ID"] == request_id
    assert search.json()["request_id"] == request_id
    assert answer.status_code == 200
    assert extraction.status_code == 200
    assert documents.status_code == 200
    assert detail.status_code == 200
    assert health.status_code == 200
    assert models.status_code == 200


def test_validation_body_limit_and_structured_not_found_errors() -> None:
    from kawaneen.api.app import create_app

    with TestClient(create_app(lambda: _container())) as client:
        validation = client.post("/v1/search", json={"query": "x", "jurisdiction": "EG"})
        oversized = client.post(
            "/v1/extract",
            content=b"x" * (128 * 1024 + 1),
            headers={"Content-Type": "application/json"},
        )
        not_found = client.get("/v1/documents/missing")

    assert validation.status_code == 422
    assert validation.json()["error"]["code"] == "VALIDATION_ERROR"
    assert validation.json()["request_id"]
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    assert not_found.status_code == 404
    assert not_found.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_timeout_and_internal_error_are_safe_and_request_ids_are_isolated() -> None:
    from kawaneen.api.app import ApiSettings, create_app

    with TestClient(
        create_app(
            lambda: _container(delay=0.05), api_settings=ApiSettings(search_timeout_seconds=0.001)
        )
    ) as client:
        timeout = client.post("/v1/search", json={"query": "q", "jurisdiction": "SA"})
        error = client.post(
            "/v1/search",
            json={"query": "q", "jurisdiction": "SA"},
            headers={"X-Request-ID": "bad id"},
        )

    assert timeout.status_code == 504
    assert timeout.json()["error"]["code"] == "REQUEST_TIMEOUT"
    assert error.status_code == 504
    assert error.headers["X-Request-ID"] != "bad id"
    assert "local path" not in error.text


def test_degraded_health_and_missing_service_return_503() -> None:
    from kawaneen.api.app import create_app

    with TestClient(create_app(lambda: ServiceContainer())) as client:
        health = client.get("/v1/health")
        search = client.post("/v1/search", json={"query": "q", "jurisdiction": "SA"})

    assert health.status_code == 503
    assert health.json()["status"] == "degraded"
    assert search.status_code == 503
    assert search.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


def test_hybrid_extraction_unavailable_is_not_a_deterministic_fallback() -> None:
    from kawaneen.api.app import create_app

    with TestClient(create_app(lambda: _container())) as client:
        response = client.post(
            "/v1/extract",
            json={"text": "يلتزم الطرف بالسداد.", "jurisdiction": "SA", "mode": "hybrid"},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_unexpected_service_error_is_a_safe_500() -> None:
    from kawaneen.api.app import create_app

    with TestClient(create_app(lambda: _container(retriever=FakeRetriever(fail=True)))) as client:
        response = client.post("/v1/search", json={"query": "q", "jurisdiction": "SA"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "local path" not in response.text
