"""Private JSONL review packets and explicit, auditable review transitions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from kawaneen.evaluation.models import (
    DatasetItem,
    ReviewMetadata,
    ReviewState,
    citation_to_dict,
    span_to_dict,
)
from kawaneen.evaluation.serialization import read_items_jsonl


class ReviewTransitionError(ValueError):
    pass


_PII = re.compile(
    r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b|"
    r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d))"
)


def _redact_review_excerpt(value: str) -> str:
    return _PII.sub("[REDACTED]", value)


def _redact_with_offsets(value: str) -> tuple[str, list[int]]:
    output: list[str] = []
    offsets = [0] * (len(value) + 1)
    cursor = 0
    output_length = 0
    for match in _PII.finditer(value):
        for index in range(cursor, match.start() + 1):
            offsets[index] = output_length + index - cursor
        output.append(value[cursor : match.start()])
        output_length += match.start() - cursor
        replacement = "[REDACTED]"
        output.append(replacement)
        output_length += len(replacement)
        for index in range(match.start(), match.end() + 1):
            offsets[index] = (
                output_length if index == match.end() else output_length - len(replacement)
            )
        cursor = match.end()
    for index in range(cursor, len(value) + 1):
        offsets[index] = output_length + index - cursor
    output.append(value[cursor:])
    return "".join(output), offsets


_ALLOWED: dict[ReviewState, set[ReviewState]] = {
    ReviewState.DRAFT: {ReviewState.PRIMARY_REVIEWED},
    ReviewState.PRIMARY_REVIEWED: {ReviewState.SECONDARY_REVIEWED, ReviewState.ADJUDICATED},
    ReviewState.SECONDARY_REVIEWED: {ReviewState.ADJUDICATED},
    ReviewState.ADJUDICATED: {ReviewState.FROZEN},
    ReviewState.FROZEN: set(),
}


def _packet_record(
    item: DatasetItem, unit_texts: dict[str, str] | None = None
) -> dict[str, object]:
    excerpts: list[dict[str, object]] = []
    for group in item.evidence_groups:
        for span in group.spans:
            text = (unit_texts or {}).get(span.unit_id, "")
            excerpt_start = max(0, span.start - 180)
            excerpt = text[excerpt_start : min(len(text), span.end + 180)]
            if text and span.start < len(text):
                local_start = min(span.start - excerpt_start, len(excerpt))
                local_end = min(span.end - excerpt_start, len(excerpt))
                excerpt, offsets = _redact_with_offsets(excerpt)
                local_start = offsets[local_start]
                local_end = offsets[local_end]
                excerpt = (
                    excerpt[:local_start]
                    + "[[EVIDENCE]]"
                    + excerpt[local_start:local_end]
                    + "[[/EVIDENCE]]"
                    + excerpt[local_end:]
                )
            else:
                excerpt = _redact_review_excerpt(excerpt)
            excerpts.append(
                {"unit_id": span.unit_id, "start": span.start, "end": span.end, "excerpt": excerpt}
            )
    item_dump = item.model_dump(mode="json")
    semantic_target = item_dump.get("semantic_target")
    if isinstance(semantic_target, dict):
        semantic_target = cast(dict[str, object], semantic_target)
        item_dump["semantic_target"] = {
            key: _redact_review_excerpt(value) if isinstance(value, str) else value
            for key, value in semantic_target.items()
        }
    return {
        "item": item_dump,
        "query_id": item.query_id,
        "intent_id": item.intent_id,
        "variant_id": item.variant_id,
        "query_text": item.query_text,
        "language": item.language.value,
        "register": item.register.value,
        "category": item.category.value,
        "query_type": item.query_type.value,
        "answerability": item.answerability.value,
        "difficulty": item.difficulty.value,
        "gold_answer": item.gold_answer,
        "source_document_ids": list(item.source_document_ids),
        "evidence": [
            {"group_id": group.group_id, "spans": [span_to_dict(span) for span in group.spans]}
            for group in item.evidence_groups
        ],
        "source_excerpts": excerpts,
        "citations": [citation_to_dict(anchor) for anchor in item.citation_anchors],
        "review": item.review.model_dump(),
        "editable_review": {
            "state": item.review.state.value,
            "reviewer_id": item.review.primary_reviewer,
            "secondary_reviewer_id": item.review.secondary_reviewer,
            "adjudicator_id": item.review.adjudicator,
            "decision": item.review.primary_decision,
            "secondary_decision": item.review.secondary_decision,
            "notes": item.review.notes,
            "human_verified": False,
            "attestation": False,
        },
    }


def export_review_packet(
    source: Path, destination: Path, unit_texts: dict[str, str] | None = None
) -> Path:
    """Export private review material; it never changes item verification state."""

    items = read_items_jsonl(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(_packet_record(item, unit_texts), ensure_ascii=False, sort_keys=True) + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    return destination


def import_review_packet(path: Path) -> tuple[DatasetItem, ...]:
    """Apply explicit review fields while refusing invalid transitions and auto-verification."""

    result: list[DatasetItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        item_data = record.get("item") or record
        current = ReviewMetadata.model_validate(item_data.get("review", {}))
        editable = record.get("editable_review", {})
        desired = ReviewState(str(editable.get("state", current.state.value)))
        if desired is not current.state and desired not in _ALLOWED[current.state]:
            raise ReviewTransitionError(
                f"invalid review transition {current.state.value} -> {desired.value}"
            )
        # Human verification requires explicit attestation; import never infers it.
        verified = bool(editable.get("human_verified", False)) and bool(
            editable.get("attestation", False)
        )
        review = current.model_copy(
            update={
                "state": desired,
                "human_verified": verified,
                "primary_reviewer": editable.get("reviewer_id") or current.primary_reviewer,
                "secondary_reviewer": editable.get("secondary_reviewer_id")
                or current.secondary_reviewer,
                "adjudicator": editable.get("adjudicator_id") or current.adjudicator,
                "primary_decision": editable.get("decision") or current.primary_decision,
                "secondary_decision": editable.get("secondary_decision")
                or current.secondary_decision,
                "notes": str(editable.get("notes", current.notes)),
            }
        )
        if "query_id" not in item_data:
            raise ReviewTransitionError("review packet record is missing item identity")
        result.append(
            DatasetItem.model_validate({**item_data, "review": review.model_dump(mode="json")})
        )
    return tuple(result)


def import_reviews(source: Path, packet: Path) -> tuple[DatasetItem, ...]:
    source_items = {item.query_id: item for item in read_items_jsonl(source)}
    result: list[DatasetItem] = []
    for line in packet.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        query_id = str(record["query_id"])
        item = source_items[query_id]
        current = item.review
        editable = record.get("editable_review", {})
        desired = ReviewState(str(editable.get("state", current.state.value)))
        if desired is not current.state and desired not in _ALLOWED[current.state]:
            raise ReviewTransitionError(
                f"invalid review transition {current.state.value} -> {desired.value}"
            )
        attested = bool(editable.get("human_verified", False)) and bool(
            editable.get("attestation", False)
        )
        updated = current.model_copy(
            update={
                "state": desired,
                "human_verified": attested,
                "primary_reviewer": editable.get("reviewer_id") or current.primary_reviewer,
                "secondary_reviewer": editable.get("secondary_reviewer_id")
                or current.secondary_reviewer,
                "adjudicator": editable.get("adjudicator_id") or current.adjudicator,
                "primary_decision": editable.get("decision") or current.primary_decision,
                "secondary_decision": editable.get("secondary_decision")
                or current.secondary_decision,
                "notes": str(editable.get("notes", current.notes)),
            }
        )
        result.append(item.model_copy(update={"review": updated}))
    return tuple(result)


def review_status(items: tuple[DatasetItem, ...]) -> dict[str, int | bool]:
    return {
        "item_count": len(items),
        "draft": sum(item.review.state is ReviewState.DRAFT for item in items),
        "primary_reviewed": sum(item.review.state is not ReviewState.DRAFT for item in items),
        "secondary_reviewed": sum(
            item.review.state
            in {ReviewState.SECONDARY_REVIEWED, ReviewState.ADJUDICATED, ReviewState.FROZEN}
            for item in items
        ),
        "human_verified": sum(item.review.human_verified for item in items),
        "unresolved_disagreements": sum(item.review.disagreement for item in items),
    }
