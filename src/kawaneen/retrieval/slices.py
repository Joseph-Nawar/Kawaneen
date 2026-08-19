# pyright: basic
"""Frozen benchmark slice assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from kawaneen.evaluation.models import Answerability, DatasetItem
from kawaneen.normalization.tokenization import tokenize


@dataclass(frozen=True, slots=True)
class QueryLengthBins:
    short_max: int
    medium_max: int

    def assign(self, query_text: str) -> str:
        length = len(tokenize(query_text))
        if length <= self.short_max:
            return "short"
        if length <= self.medium_max:
            return "medium"
        return "long"

    def to_dict(self) -> dict[str, int]:
        return {"short_max": self.short_max, "medium_max": self.medium_max}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> QueryLengthBins:
        return cls(
            short_max=int(cast(int, payload["short_max"])),
            medium_max=int(cast(int, payload["medium_max"])),
        )


def build_query_length_bins(items: Sequence[DatasetItem]) -> QueryLengthBins:
    lengths = sorted(
        len(tokenize(item.query_text))
        for item in items
        if item.answerability is Answerability.ANSWERABLE and item.variant_id is None
    )
    if not lengths:
        raise ValueError("answerable base queries are required for length bins")
    short_index = max(0, (len(lengths) + 2) // 3 - 1)
    medium_index = max(short_index, (2 * len(lengths) + 2) // 3 - 1)
    return QueryLengthBins(short_max=lengths[short_index], medium_max=lengths[medium_index])


def _evidence_type(item: DatasetItem, unit_type_by_id: Mapping[str, str]) -> str:
    families: set[str] = set()
    for group in item.evidence_groups:
        for span in group.spans:
            unit_type = unit_type_by_id.get(span.unit_id, "")
            if unit_type in {"facts", "events"}:
                families.add("facts/events")
            elif unit_type == "reasoning":
                families.add("reasoning")
            elif unit_type in {"ruling", "verdict"}:
                families.add("ruling/verdict")
            elif unit_type == "applicable_laws":
                families.add("applicable_laws")
            else:
                families.add("unknown")
    return next(iter(families)) if len(families) == 1 and "unknown" not in families else "mixed"


def assign_slices(
    item: DatasetItem,
    query_length_bins: QueryLengthBins,
    unit_type_by_id: Mapping[str, str],
    source_by_document: Mapping[str, str],
) -> dict[str, str]:
    sources = {
        source_by_document[doc] for doc in item.source_document_ids if doc in source_by_document
    }
    source = next(iter(sources)) if len(sources) == 1 else "mixed"
    evidence_type = _evidence_type(item, unit_type_by_id)
    return {
        "category": item.category.value,
        "question_type": item.query_type.value,
        "language": item.language.value,
        "register": item.register.value,
        "gold_evidence_type": evidence_type,
        "document_type": evidence_type,
        "difficulty": item.difficulty.value,
        "query_length": query_length_bins.assign(item.query_text),
        "jurisdiction": item.jurisdiction,
        "source": source,
        "base_vs_variant": "variant" if item.variant_id is not None else "base",
    }
