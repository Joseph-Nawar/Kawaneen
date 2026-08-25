"""Fail-closed exact source span resolution."""

from __future__ import annotations

from kawaneen.extraction.contracts import ExactSourceSpan


def resolve_exact_span(
    canonical_text: str,
    proposed_text: str,
    *,
    occurrence: int | None = None,
    canonical_unit_id: str = "synthetic-unit",
    document_id: str = "synthetic-document",
) -> ExactSourceSpan:
    if not proposed_text:
        raise ValueError("proposed source span is empty")
    offsets: list[int] = []
    cursor = 0
    while True:
        found = canonical_text.find(proposed_text, cursor)
        if found < 0:
            break
        offsets.append(found)
        cursor = found + 1
    if not offsets:
        raise ValueError("proposed source span not found in canonical text")
    if occurrence is None and len(offsets) != 1:
        raise ValueError("proposed source span is ambiguous")
    selected_index = 0 if occurrence is None else occurrence
    if selected_index < 0 or selected_index >= len(offsets):
        raise ValueError("source span occurrence is out of range")
    start = offsets[selected_index]
    return ExactSourceSpan(
        text=proposed_text,
        start_char=start,
        end_char=start + len(proposed_text),
        canonical_unit_id=canonical_unit_id,
        document_id=document_id,
    )


def validate_exact_span(canonical_text: str, span: ExactSourceSpan) -> bool:
    return canonical_text[span.start_char : span.end_char] == span.text
