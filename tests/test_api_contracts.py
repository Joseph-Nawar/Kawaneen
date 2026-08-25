from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_search_request_is_strict_and_bounded() -> None:
    from kawaneen.api.contracts import SearchRequest

    request = SearchRequest(query="  ما هي المهلة؟  ", jurisdiction="SA", limit=3)
    assert request.query == "ما هي المهلة؟"
    assert request.limit == 3

    with pytest.raises(ValidationError):
        SearchRequest(query="x", jurisdiction="SA", unexpected=True)
    with pytest.raises(ValidationError):
        SearchRequest(query="x" * 2001, jurisdiction="SA")
    with pytest.raises(ValidationError):
        SearchRequest(query="x", jurisdiction="EG")


def test_extract_request_has_mode_and_text_limit() -> None:
    from kawaneen.api.contracts import ExtractionMode, ExtractRequest

    assert ExtractRequest(text="نص", jurisdiction="SA").mode is ExtractionMode.HYBRID
    with pytest.raises(ValidationError):
        ExtractRequest(text="x" * 20_001, jurisdiction="SA")


def test_request_id_accepts_safe_values_only() -> None:
    from kawaneen.api.context import normalize_request_id

    assert normalize_request_id("client-123") == "client-123"
    generated = normalize_request_id("bad value")
    assert len(generated) == 36
    assert normalize_request_id("x" * 129) != "x" * 129


def test_error_envelope_is_strict() -> None:
    from kawaneen.api.contracts import ErrorResponse

    payload = ErrorResponse(
        error={"code": "DOCUMENT_NOT_FOUND", "message": "document not found"},
        request_id="req-1",
    )
    assert payload.error.code == "DOCUMENT_NOT_FOUND"
    with pytest.raises(ValidationError):
        ErrorResponse(
            error={"code": "INTERNAL_ERROR", "message": "x", "debug": "secret"},
            request_id="req-1",
        )
