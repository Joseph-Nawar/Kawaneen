"""Frozen retrieval-candidate selection over Phase 3 canonical units."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pyarrow.parquet as _parquet

from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType

pq: Any = cast(Any, _parquet)

CONTENT_UNIT_TYPES = tuple(
    sorted(
        {
            UnitType.APPLICABLE_LAWS.value,
            UnitType.CASE_TEXT.value,
            UnitType.COURT_REASONING.value,
            UnitType.EVENTS.value,
            UnitType.FACTS.value,
            UnitType.REASONING.value,
            UnitType.RULING.value,
            UnitType.VERDICT.value,
        }
    )
)
ELIGIBLE_SOURCES = ("alarb", "arabiccr")


@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    schema_version: int
    sources: tuple[str, ...]
    source_versions: Mapping[str, str]
    included_unit_types: tuple[str, ...]
    exclusion_rules: tuple[str, ...]
    source_counts: Mapping[str, int]
    unit_type_counts: Mapping[str, int]
    candidate_count: int
    candidate_ids: frozenset[str]
    candidate_ids_hash: str
    manifest_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_versions", MappingProxyType(dict(self.source_versions)))
        object.__setattr__(self, "source_counts", MappingProxyType(dict(self.source_counts)))
        object.__setattr__(self, "unit_type_counts", MappingProxyType(dict(self.unit_type_counts)))

    def to_sanitized_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sources": list(self.sources),
            "source_versions": dict(self.source_versions),
            "included_unit_types": list(self.included_unit_types),
            "exclusion_rules": list(self.exclusion_rules),
            "source_counts": dict(self.source_counts),
            "unit_type_counts": dict(self.unit_type_counts),
            "candidate_count": self.candidate_count,
            "candidate_ids_hash": self.candidate_ids_hash,
            "manifest_hash": self.manifest_hash,
        }


def _candidate_from_row(row: dict[str, object]) -> CanonicalUnit:
    return CanonicalUnit(
        unit_id=str(row["unit_id"]),
        document_id=str(row["document_id"]),
        unit_type=UnitType(str(row["unit_type"])),
        text=str(row["text"]),
        provenance=SourceProvenance(
            source_id=str(row["source_id"]),
            source_version=str(row["source_version"]),
            source_path=str(row["source_path"]),
            source_row=cast(int, row["source_row"]),
            source_field=str(row["source_field"]),
            split=str(row.get("split") or ""),
        ),
        ordinal=cast(int, row["ordinal"]) if row.get("ordinal") is not None else None,
    )


def load_candidate_units(
    root: Path, sources: tuple[str, ...] = ELIGIBLE_SOURCES
) -> tuple[CanonicalUnit, ...]:
    """Load exactly one frozen canonical version per eligible source."""

    candidates: list[CanonicalUnit] = []
    for source in sorted(sources):
        source_root = root / source
        versions = sorted(path for path in source_root.iterdir() if path.is_dir())
        if len(versions) != 1:
            raise ValueError(f"expected exactly one canonical version for {source}")
        table: Any = pq.read_table(versions[0] / "units.parquet")
        rows = cast(list[dict[str, object]], table.to_pylist())
        for raw_row in rows:
            row = dict(raw_row)
            if str(row["unit_type"]) not in CONTENT_UNIT_TYPES:
                continue
            if not str(row["text"]).strip():
                continue
            candidates.append(_candidate_from_row(row))
    return tuple(
        sorted(
            candidates,
            key=lambda unit: (
                unit.provenance.source_id,
                unit.provenance.source_version,
                unit.provenance.source_row,
                unit.unit_id,
            ),
        )
    )


def select_representative_subset(
    units: Iterable[CanonicalUnit], *, per_source_unit_type: int = 1500
) -> tuple[CanonicalUnit, ...]:
    """Select a stable balanced subset before any retrieval result is observed."""

    if per_source_unit_type < 1:
        raise ValueError("per_source_unit_type must be positive")
    groups: defaultdict[tuple[str, str], list[CanonicalUnit]] = defaultdict(list)
    for unit in units:
        groups[(unit.provenance.source_id, unit.unit_type.value)].append(unit)
    selected: list[CanonicalUnit] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda unit: (unit.unit_id, unit.provenance.source_row))
        if len(group) <= per_source_unit_type:
            selected.extend(group)
            continue
        last = per_source_unit_type - 1
        denominator = len(group) - 1
        selected.extend(
            group[(index * denominator) // last] for index in range(per_source_unit_type)
        )
    return tuple(
        sorted(
            selected,
            key=lambda unit: (
                unit.provenance.source_id,
                unit.provenance.source_version,
                unit.provenance.source_row,
                unit.unit_id,
            ),
        )
    )


def freeze_candidate_policy(units: Iterable[CanonicalUnit]) -> CandidatePolicy:
    selected = tuple(units)
    source_versions = {
        unit.provenance.source_id: unit.provenance.source_version for unit in selected
    }
    source_counts = Counter(unit.provenance.source_id for unit in selected)
    unit_type_counts = Counter(unit.unit_type.value for unit in selected)
    candidate_ids = tuple(unit.unit_id for unit in selected)
    candidate_ids_hash = hashlib.sha256(",".join(candidate_ids).encode("utf-8")).hexdigest()
    exclusion_rules = (
        "include only content-bearing canonical unit types",
        "exclude empty or whitespace-only text",
        "exclude OCR-derived text and metadata fields",
    )
    payload = {
        "schema_version": 1,
        "sources": sorted(source_versions),
        "source_versions": dict(sorted(source_versions.items())),
        "included_unit_types": sorted(unit_type_counts),
        "exclusion_rules": list(exclusion_rules),
        "source_counts": dict(sorted(source_counts.items())),
        "unit_type_counts": dict(sorted(unit_type_counts.items())),
        "candidate_count": len(selected),
        "candidate_ids_hash": candidate_ids_hash,
    }
    manifest_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return CandidatePolicy(
        schema_version=1,
        sources=tuple(sorted(source_versions)),
        source_versions=source_versions,
        included_unit_types=tuple(sorted(unit_type_counts)),
        exclusion_rules=exclusion_rules,
        source_counts=source_counts,
        unit_type_counts=unit_type_counts,
        candidate_count=len(selected),
        candidate_ids=frozenset(candidate_ids),
        candidate_ids_hash=candidate_ids_hash,
        manifest_hash=manifest_hash,
    )
