# pyright: basic, reportOptionalOperand=false
"""Explicit structured metadata filters and coverage accounting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

FIELDS = (
    "jurisdiction",
    "issuing_authority",
    "document_type",
    "publication_date",
    "legal_status",
    "regulation_name",
)


def _date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid ISO date constraint: {value}") from exc


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    document_id: str
    jurisdiction: str | None = None
    issuing_authority: str | None = None
    document_type: str | None = None
    publication_date: date | None = None
    legal_status: str | None = None
    regulation_name: str | None = None


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    jurisdiction: tuple[str, ...] | None = None
    issuing_authority: tuple[str, ...] | None = None
    document_type: tuple[str, ...] | None = None
    publication_date_from: date | str | None = None
    publication_date_to: date | str | None = None
    legal_status: tuple[str, ...] | None = None
    regulation_name: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for field in (
            "jurisdiction",
            "issuing_authority",
            "document_type",
            "legal_status",
            "regulation_name",
        ):
            values = getattr(self, field)
            if values is not None and (
                not isinstance(values, tuple)
                or len(values) == 0
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(f"{field} contains an empty or invalid value")
        start = _date(self.publication_date_from)
        end = _date(self.publication_date_to)
        if start and end and start > end:
            raise ValueError("publication date range is inverted")
        object.__setattr__(self, "publication_date_from", start)
        object.__setattr__(self, "publication_date_to", end)

    @property
    def explicit(self) -> bool:
        return any(
            (
                self.jurisdiction,
                self.issuing_authority,
                self.document_type,
                self.publication_date_from,
                self.publication_date_to,
                self.legal_status,
                self.regulation_name,
            )
        )


@dataclass(frozen=True, slots=True)
class MetadataIndex:
    records: dict[str, DocumentMetadata]

    @classmethod
    def build(cls, records: Iterable[DocumentMetadata]) -> MetadataIndex:
        values = tuple(records)
        ids = [record.document_id for record in values]
        if len(ids) != len(set(ids)):
            raise ValueError("metadata document IDs must be unique")
        return cls({record.document_id: record for record in values})

    def eligible_ids(self, constraint: MetadataFilter) -> set[str]:
        return {
            document_id
            for document_id, record in self.records.items()
            if _matches(record, constraint)
        }


def _matches(record: DocumentMetadata, constraint: MetadataFilter) -> bool:
    for field in (
        "jurisdiction",
        "issuing_authority",
        "document_type",
        "legal_status",
        "regulation_name",
    ):
        allowed = getattr(constraint, field)
        if allowed is not None and getattr(record, field) not in allowed:
            return False
    start = _date(constraint.publication_date_from)
    end = _date(constraint.publication_date_to)
    if (start or end) and record.publication_date is None:
        return False
    if start and record.publication_date < start:
        return False
    return not (end and record.publication_date > end)


def metadata_coverage(
    records: Iterable[DocumentMetadata], *, expected_document_ids: Iterable[str] | None = None
) -> dict[str, object]:
    by_id = {record.document_id: record for record in records}
    ids = tuple(expected_document_ids) if expected_document_ids is not None else tuple(by_id)
    fields: dict[str, object] = {}
    for field in FIELDS:
        values = [
            getattr(by_id[document_id], field, None) for document_id in ids if document_id in by_id
        ]
        populated = [str(value) for value in values if value is not None and str(value)]
        distinct = Counter(populated)
        fields[field] = {
            "populated_count": len(populated),
            "null_count": len(ids) - len(populated),
            "populated_percentage": len(populated) / max(len(ids), 1),
            "distinct_value_count": len(distinct),
            "distinct_values": sorted(distinct),
        }
    return {"schema_version": 1, "document_count": len(ids), "fields": fields}
