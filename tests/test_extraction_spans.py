import pytest

from kawaneen.extraction.span_validation import resolve_exact_span, validate_exact_span


def test_exact_span_is_accepted() -> None:
    span = resolve_exact_span("يجب على المنشأة التسجيل.", "المنشأة")
    assert span.start_char == 8
    assert span.end_char == 15
    assert span.text == "المنشأة"


def test_nonexistent_span_is_rejected() -> None:
    with pytest.raises(ValueError, match="not found"):
        resolve_exact_span("يجب التسجيل.", "المنشأة")


def test_ambiguous_span_fails_closed_without_occurrence() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_exact_span("المادة 1 والمادة 1", "المادة 1")


def test_occurrence_contract_can_select_one_exact_occurrence() -> None:
    span = resolve_exact_span("المادة 1 والمادة 1", "المادة 1", occurrence=1)
    assert span.start_char == 10
    assert validate_exact_span("المادة 1 والمادة 1", span)
