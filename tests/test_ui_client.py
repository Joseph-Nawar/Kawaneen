from __future__ import annotations

import httpx
import pytest

from kawaneen.ui.client import HttpUiClient, UiApiError


def _search_payload() -> dict[str, object]:
    return {
        "request_id": "req-1",
        "jurisdiction": "SA",
        "results": [
            {
                "chunk_id": "chunk-1",
                "rank": 1,
                "text": "يلتزم الطرف بالسداد خلال ثلاثين يوماً.",
                "document_id": "doc-1",
                "document_title": "نظام تجريبي",
                "article": "المادة 1",
                "page": "1",
                "source_url": None,
                "score": 2.1,
                "score_type": "reranker_raw_logit",
                "provenance": "both",
            }
        ],
        "retrieval": {
            "strategy": "hybrid_reranked",
            "sparse_top_k": 50,
            "dense_top_k": 50,
            "fused_candidate_count": 20,
            "reranker_depth": 8,
            "top_score": 2.1,
            "hit_count": 1,
            "returned_count": 1,
            "score_type": "reranker_raw_logit",
        },
        "latency_ms": 12.5,
        "warnings": [],
    }


def test_client_validates_successful_phase12_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/search"
        assert request.headers["X-Request-ID"]
        return httpx.Response(200, json=_search_payload())

    client = HttpUiClient("http://api.test", transport=httpx.MockTransport(handler))

    result = client.search(query="مدة الاعتراض", limit=4)

    assert result.results[0].document_id == "doc-1"
    assert result.retrieval.score_type == "reranker_raw_logit"


def test_client_maps_safe_api_error_without_exposing_payload() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "error": {"code": "SERVICE_UNAVAILABLE", "message": "retrieval unavailable"},
                "request_id": "req-2",
                "private": "/machine/path",
            },
        )

    client = HttpUiClient("http://api.test", transport=httpx.MockTransport(handler))

    with pytest.raises(UiApiError) as error:
        client.search(query="query", limit=8)

    assert error.value.code == "SERVICE_UNAVAILABLE"
    assert error.value.request_id == "req-2"
    assert "/machine/path" not in str(error.value)


def test_client_maps_transport_failure_to_degraded_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret provider payload", request=_)

    client = HttpUiClient("http://api.test", transport=httpx.MockTransport(handler))

    with pytest.raises(UiApiError) as error:
        client.health()

    assert error.value.code == "API_UNAVAILABLE"
    assert error.value.message == "The Phase 12 API could not be reached."
