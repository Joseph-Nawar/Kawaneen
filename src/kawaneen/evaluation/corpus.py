"""Full governed retrieval-corpus scope for Phase 6."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.normalization.corpus import load_candidate_units

CONTENT_POLICY_VERSION = "phase5-source-content-policy-v1"
ELIGIBLE_SOURCES = ("alarb", "arabiccr")
ALARB_TYPES = frozenset({"facts", "court_reasoning", "applicable_laws", "verdict"})
ARABICCR_STRUCTURED_TYPES = frozenset({"events", "reasoning", "ruling"})


@dataclass(frozen=True, slots=True)
class EvaluationCorpus:
    units: tuple[CanonicalUnit, ...]
    document_ids: frozenset[str]
    unit_ids: frozenset[str]
    document_count_by_source: Mapping[str, int]
    unit_count_by_source: Mapping[str, int]
    unit_type_counts: Mapping[str, int]
    source_versions: Mapping[str, str]
    canonical_hashes: Mapping[str, str]
    content_policy_version: str
    content_policy_hash: str
    document_ids_hash: str
    unit_ids_hash: str
    corpus_hash: str

    def __post_init__(self) -> None:
        for name in (
            "document_count_by_source",
            "unit_count_by_source",
            "unit_type_counts",
            "source_versions",
            "canonical_hashes",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def document_count(self) -> int:
        return len(self.document_ids)

    @property
    def unit_count(self) -> int:
        return len(self.units)


def _hash_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_hashes(root: Path, inventory_path: Path) -> dict[str, str]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for source in inventory["sources"]:
        if source["source_id"] not in ELIGIBLE_SOURCES:
            continue
        for file in source["files"]:
            relative = Path(str(file["path"]))
            path = relative if relative.is_absolute() else Path.cwd() / relative
            if not path.is_file():
                raise ValueError(f"canonical file is missing: {relative}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != file["sha256"]:
                raise ValueError(f"canonical hash mismatch: {relative}")
            result[relative.as_posix()] = actual
    return dict(sorted(result.items()))


def load_evaluation_units(root: Path) -> tuple[CanonicalUnit, ...]:
    """Load the complete Phase-3 candidate corpus and apply Phase-5 source policy."""

    candidates = load_candidate_units(root, sources=ELIGIBLE_SOURCES)
    by_document: defaultdict[tuple[str, str], list[CanonicalUnit]] = defaultdict(list)
    for unit in candidates:
        by_document[(unit.provenance.source_id, unit.document_id)].append(unit)
    selected: list[CanonicalUnit] = []
    for (source, _document_id), rows in sorted(by_document.items()):
        if source == "alarb":
            selected.extend(unit for unit in rows if unit.unit_type.value in ALARB_TYPES)
            continue
        structured = any(
            unit.unit_type.value in ARABICCR_STRUCTURED_TYPES and unit.text.strip() for unit in rows
        )
        selected.extend(
            unit
            for unit in rows
            if unit.unit_type.value in ARABICCR_STRUCTURED_TYPES
            or (unit.unit_type.value == "case_text" and not structured)
        )
    return tuple(
        sorted(
            selected,
            key=lambda unit: (
                unit.provenance.source_id,
                unit.provenance.source_version,
                unit.provenance.source_row,
                unit.ordinal or 0,
                unit.unit_id,
            ),
        )
    )


def freeze_evaluation_corpus(
    units: Iterable[CanonicalUnit],
    *,
    canonical_root: Path,
    inventory_path: Path = Path("data/manifests/canonical/inventory.json"),
) -> EvaluationCorpus:
    selected = tuple(units)
    if not selected:
        raise ValueError("evaluation corpus cannot be empty")
    if {unit.provenance.source_id for unit in selected} - set(ELIGIBLE_SOURCES):
        raise ValueError("evaluation corpus sources must be eligible ALARB/ArabiCCR sources")
    allowed = ALARB_TYPES | ARABICCR_STRUCTURED_TYPES | {"case_text"}
    if any(unit.unit_type.value not in allowed for unit in selected):
        raise ValueError("evaluation corpus contains an ineligible unit type")
    for unit in selected:
        if unit.provenance.source_id == "arabiccr" and unit.unit_type.value == "case_text":
            siblings = [other for other in selected if other.document_id == unit.document_id]
            if any(other.unit_type.value in ARABICCR_STRUCTURED_TYPES for other in siblings):
                raise ValueError("case_text fallback cannot coexist with structured ArabiCCR units")
    document_ids = frozenset(unit.document_id for unit in selected)
    unit_ids = frozenset(unit.unit_id for unit in selected)
    if len(unit_ids) != len(selected):
        raise ValueError("evaluation corpus contains duplicate unit IDs")
    source_versions = {
        source: next(
            unit.provenance.source_version
            for unit in selected
            if unit.provenance.source_id == source
        )
        for source in sorted({unit.provenance.source_id for unit in selected})
    }
    canonical_hashes = _canonical_hashes(canonical_root, inventory_path)
    policy = {
        "version": CONTENT_POLICY_VERSION,
        "sources": ELIGIBLE_SOURCES,
        "alarb_unit_types": sorted(ALARB_TYPES),
        "arabiccr_structured_unit_types": sorted(ARABICCR_STRUCTURED_TYPES),
        "arabiccr_fallback": "case_text only when structured content is unavailable",
        "exclude": ["ocr-derived material", "saudi-moj-derived statute seed"],
    }
    policy_hash = _hash_json(policy)
    document_counts = Counter(unit.provenance.source_id for unit in selected)
    documents_by_source = {
        source: len({unit.document_id for unit in selected if unit.provenance.source_id == source})
        for source in sorted(source_versions)
    }
    type_counts = Counter(unit.unit_type.value for unit in selected)
    ordered_document_ids = tuple(sorted(document_ids))
    ordered_unit_ids = tuple(sorted(unit_ids))
    document_ids_hash = hashlib.sha256(",".join(ordered_document_ids).encode()).hexdigest()
    unit_ids_hash = hashlib.sha256(",".join(ordered_unit_ids).encode()).hexdigest()
    corpus_hash = _hash_json(
        {
            "canonical_hashes": canonical_hashes,
            "content_policy_hash": policy_hash,
            "document_ids_hash": document_ids_hash,
            "unit_ids_hash": unit_ids_hash,
            "source_versions": source_versions,
        }
    )
    return EvaluationCorpus(
        units=selected,
        document_ids=document_ids,
        unit_ids=unit_ids,
        document_count_by_source=documents_by_source,
        unit_count_by_source=dict(document_counts),
        unit_type_counts=dict(type_counts),
        source_versions=source_versions,
        canonical_hashes=canonical_hashes,
        content_policy_version=CONTENT_POLICY_VERSION,
        content_policy_hash=policy_hash,
        document_ids_hash=document_ids_hash,
        unit_ids_hash=unit_ids_hash,
        corpus_hash=corpus_hash,
    )


def corpus_summary(corpus: EvaluationCorpus) -> dict[str, object]:
    """Return only text-free corpus metadata suitable for tracking."""

    return {
        "schema_version": 1,
        "status": "phase6_corpus_scope_frozen_before_candidate_generation",
        "content_policy_version": corpus.content_policy_version,
        "content_policy_hash": corpus.content_policy_hash,
        "source_versions": dict(corpus.source_versions),
        "canonical_hashes": dict(corpus.canonical_hashes),
        "document_count": corpus.document_count,
        "document_count_by_source": dict(corpus.document_count_by_source),
        "unit_count": corpus.unit_count,
        "unit_count_by_source": dict(corpus.unit_count_by_source),
        "unit_type_counts": dict(sorted(corpus.unit_type_counts.items())),
        "document_ids_hash": corpus.document_ids_hash,
        "unit_ids_hash": corpus.unit_ids_hash,
        "corpus_hash": corpus.corpus_hash,
        "ocr_included": False,
        "moj_retrieval_gold": False,
    }
