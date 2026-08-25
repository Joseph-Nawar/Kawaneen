"""Pure presentation helpers for evidence and extraction findings."""

from __future__ import annotations

import html
from collections.abc import Iterable

from kawaneen.api.contracts import DocumentUnit, Evidence, ExtractionResponse
from kawaneen.ui.formatting import locate_quote, text_direction


def filter_returned_evidence(
    results: Iterable[Evidence], selected_documents: set[str]
) -> tuple[Evidence, ...]:
    """Filter only API-returned evidence while preserving its original order/ranks."""

    returned = tuple(results)
    if not selected_documents:
        return returned
    return tuple(
        item
        for item in returned
        if item.document_id in selected_documents
        or (item.document_title and item.document_title in selected_documents)
    )


def inspect_verified_quote(unit: DocumentUnit, quote: str) -> str:
    """Render one canonical unit with an exact, safely escaped quote highlight."""

    location = locate_quote((unit,), quote)
    if location is None:
        return ""
    before = html.escape(unit.text[: location.start_char])
    matched = html.escape(unit.text[location.start_char : location.end_char])
    after = html.escape(unit.text[location.end_char :])
    direction = text_direction(unit.text)
    metadata = html.escape(
        f"canonical unit: {unit.unit_id} · type: {unit.unit_type} · "
        f"article: {unit.heading_path[-1] if unit.heading_path else 'not available'}"
    )
    return (
        f'<div class="kw-quote kw-{direction}" dir="{direction}">'
        f'<div class="kw-meta">{metadata}</div>'
        f'<div style="margin-top:.45rem;line-height:1.9">{before}'
        f'<mark class="verified-quote">{matched}</mark>{after}</div></div>'
    )


def document_page_bounds(offset: int, limit: int, total: int) -> tuple[int, int, int, bool, bool]:
    """Return inclusive visible bounds plus previous/next availability."""

    if total <= 0:
        return 0, 0, 0, False, False
    start = min(max(offset, 0), total - 1) + 1
    end = min(start + max(limit, 1) - 1, total)
    return start, end, total, offset > 0, end < total


def extract_presentation_rows(
    segment_id: str, response: ExtractionResponse
) -> tuple[dict[str, object], ...]:
    """Flatten structured findings without discarding exact spans or provenance."""

    result = response.result
    rows: list[dict[str, object]] = [
        {
            "segment_id": segment_id,
            "field": "summary",
            "value": {
                "obligations": len(result.obligations),
                "prohibitions": len(result.prohibitions),
                "permissions": len(result.permissions),
                "deadlines": len(result.deadlines),
                "regulated_entities": len(result.regulated_entities),
                "exceptions": len(result.exceptions),
            },
        }
    ]
    for rule in result.rules:
        rows.append(
            {
                "segment_id": segment_id,
                "field": "obligation" if rule in result.obligations else "rule",
                "value": {
                    "modality": rule.modality.value,
                    "actor": rule.actor_span.text if rule.actor_span else None,
                    "action": rule.action_span.text,
                    "conditions": [span.text for span in rule.condition_spans],
                    "exceptions": [span.text for span in rule.exception_spans],
                },
            }
        )
    for candidate in result.deadlines:
        rows.append(
            {
                "segment_id": segment_id,
                "field": "deadline",
                "value": {
                    "text": candidate.raw_exact_text,
                    "candidate_id": candidate.candidate_id,
                    "type": candidate.candidate_type.value,
                    "normalized": candidate.normalized.normalized_value,
                },
            }
        )
    for span in result.regulated_entities:
        rows.append({"segment_id": segment_id, "field": "regulated_entity", "value": span.text})
    for span in result.exceptions:
        rows.append({"segment_id": segment_id, "field": "exception", "value": span.text})
    provenance = result.source_provenance
    rows.append(
        {
            "segment_id": segment_id,
            "field": "source",
            "value": {
                "source_id": provenance.source_id,
                "source_version": provenance.source_version,
                "document_id": result.candidate_registry.document_id
                if result.candidate_registry
                else None,
                "fingerprint": result.source_fingerprint,
            },
        }
    )
    return tuple(rows)
