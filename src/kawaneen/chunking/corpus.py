"""Document-level Phase 5 corpus freezing over immutable canonical units."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.normalization.corpus import ELIGIBLE_SOURCES, load_candidate_units


@dataclass(frozen=True, slots=True)
class Phase5Corpus:
    units: tuple[CanonicalUnit, ...]
    document_ids: frozenset[str]
    document_count_by_source: Mapping[str, int]
    source_versions: Mapping[str, str]
    document_ids_hash: str
    scope_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "document_count_by_source", MappingProxyType(dict(self.document_count_by_source))
        )
        object.__setattr__(self, "source_versions", MappingProxyType(dict(self.source_versions)))


def load_phase5_units(root: Path = Path("data/interim/canonical")) -> tuple[CanonicalUnit, ...]:
    return load_candidate_units(root)


def freeze_phase5_documents(
    units: Iterable[CanonicalUnit], *, per_source: int = 1500
) -> Phase5Corpus:
    if per_source < 1:
        raise ValueError("per_source must be positive")
    selected_units = tuple(units)
    sources = {unit.provenance.source_id for unit in selected_units}
    if not sources.issubset(set(ELIGIBLE_SOURCES)):
        raise ValueError("Phase 5 corpus sources must be eligible ALARB/ArabiCCR sources")
    grouped: dict[str, dict[str, list[CanonicalUnit]]] = {source: {} for source in sorted(sources)}
    for unit in selected_units:
        if not unit.text.strip():
            continue
        grouped.setdefault(unit.provenance.source_id, {}).setdefault(unit.document_id, []).append(
            unit
        )
    chosen_documents: dict[str, tuple[str, ...]] = {}
    for source in sorted(sources):
        document_ids = tuple(sorted(grouped[source]))[:per_source]
        if len(document_ids) < per_source:
            raise ValueError(f"{source} has fewer than {per_source} complete documents")
        chosen_documents[source] = document_ids
    chosen_ids = {document_id for values in chosen_documents.values() for document_id in values}
    chosen = tuple(
        sorted(
            (unit for unit in selected_units if unit.document_id in chosen_ids),
            key=lambda unit: (
                unit.provenance.source_id,
                unit.provenance.source_row,
                unit.ordinal or 0,
                unit.unit_id,
            ),
        )
    )
    if {unit.document_id for unit in chosen} != chosen_ids:
        raise ValueError("frozen document scope lost a selected child")
    document_ids_hash = hashlib.sha256(",".join(sorted(chosen_ids)).encode("utf-8")).hexdigest()
    source_versions = {
        source: next(
            unit.provenance.source_version for unit in chosen if unit.provenance.source_id == source
        )
        for source in sorted(sources)
    }
    counts = Counter(unit.provenance.source_id for unit in chosen)
    payload = {
        "document_ids_hash": document_ids_hash,
        "document_count_by_source": {source: len(ids) for source, ids in chosen_documents.items()},
        "source_versions": source_versions,
        "unit_count_by_source": dict(sorted(counts.items())),
    }
    scope_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Phase5Corpus(
        units=chosen,
        document_ids=frozenset(chosen_ids),
        document_count_by_source={source: len(ids) for source, ids in chosen_documents.items()},
        source_versions=source_versions,
        document_ids_hash=document_ids_hash,
        scope_hash=scope_hash,
    )
