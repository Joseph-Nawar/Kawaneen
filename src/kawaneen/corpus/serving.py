"""Read-only, path-free canonical corpus repository for serving."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kawaneen.corpus.models import CanonicalUnit


@dataclass(frozen=True, slots=True)
class ServingUnit:
    unit_id: str
    unit_type: str
    text: str
    ordinal: int | None = None
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ServingDocument:
    document_id: str
    title: str
    source_id: str | None
    units: tuple[ServingUnit, ...]
    jurisdiction: str | None = "SA"


@dataclass(frozen=True, slots=True)
class ServingDocumentPage:
    items: tuple[ServingDocument, ...]
    offset: int
    limit: int
    total: int


class InMemoryCorpusRepository:
    def __init__(self, documents: tuple[ServingDocument, ...]) -> None:
        self._documents = tuple(sorted(documents, key=lambda item: item.document_id))

    def list_documents(self, *, offset: int = 0, limit: int = 20) -> ServingDocumentPage:
        if offset < 0 or limit < 1 or limit > 100:
            raise ValueError("invalid document pagination")
        total = len(self._documents)
        return ServingDocumentPage(
            items=self._documents[offset : offset + limit],
            offset=offset,
            limit=limit,
            total=total,
        )

    def get_document(self, document_id: str) -> ServingDocument | None:
        return next((item for item in self._documents if item.document_id == document_id), None)


class CanonicalCorpusRepository(InMemoryCorpusRepository):
    """Load canonical units once and expose only safe document/unit fields."""

    @classmethod
    def from_json(
        cls,
        canonical_units_path: Path,
        *,
        document_metadata: Mapping[str, Mapping[str, str | None]] | None = None,
    ) -> CanonicalCorpusRepository:
        payload = json.loads(canonical_units_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("canonical corpus units are invalid")
        payload_mapping = cast(dict[str, object], payload)
        if not isinstance(payload_mapping.get("units"), list):
            raise ValueError("canonical corpus units are invalid")
        grouped: dict[str, list[ServingUnit]] = {}
        source_ids: dict[str, str] = {}
        for raw in cast(list[object], payload_mapping["units"]):
            if not isinstance(raw, dict):
                raise ValueError("canonical corpus unit is invalid")
            value = dict(cast(dict[str, object], raw))
            raw_heading_path = value.pop("heading_path", [])
            heading_path = tuple(str(item) for item in cast(list[object], raw_heading_path))
            unit = CanonicalUnit.model_validate(value)
            grouped.setdefault(unit.document_id, []).append(
                ServingUnit(
                    unit_id=unit.unit_id,
                    unit_type=unit.unit_type.value,
                    text=unit.text,
                    ordinal=unit.ordinal,
                    heading_path=heading_path,
                )
            )
            source_ids[unit.document_id] = unit.provenance.source_id
        metadata = document_metadata or {}
        documents = tuple(
            ServingDocument(
                document_id=document_id,
                title=(metadata.get(document_id, {}).get("title") or ""),
                source_id=metadata.get(document_id, {}).get("source_id")
                or source_ids.get(document_id),
                units=tuple(sorted(units, key=lambda item: (item.ordinal or 0, item.unit_id))),
                jurisdiction=metadata.get(document_id, {}).get("jurisdiction") or "SA",
            )
            for document_id, units in grouped.items()
        )
        return cls(documents)


__all__ = [
    "CanonicalCorpusRepository",
    "InMemoryCorpusRepository",
    "ServingDocument",
    "ServingDocumentPage",
    "ServingUnit",
]
