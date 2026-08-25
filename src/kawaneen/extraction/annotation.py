"""Private annotation-pack preparation and fail-closed annotation validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow.parquet as _parquet
from pydantic import Field

from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.corpus.statutory import parse_article_label
from kawaneen.extraction.artifacts import write_private_json, write_text_free_json
from kawaneen.extraction.candidates import CANDIDATE_REGISTRY_VERSION, build_candidate_registry
from kawaneen.extraction.contracts import (
    CandidateRegistry,
    ExtractionModel,
    SemanticProposal,
)
from kawaneen.extraction.source_policy import eligible_regulatory_unit
from kawaneen.extraction.span_validation import resolve_exact_span
from kawaneen.sources.registry import load_registry

pq: Any = cast(Any, _parquet)

ANNOTATION_ROOT = Path("artifacts/private/phase11_extraction/annotations")
SELECTION_MANIFEST_PATH = Path("data/manifests/extraction/phase11_annotation_selection.json")
CANONICAL_ROOT = Path("data/interim/canonical")
_SEED = "phase11a-regulatory-selection-v2"
PHASE11_SELECTION_VERSION = "phase11-selection-v2"
PHASE11_ELIGIBILITY_POLICY_VERSION = "phase11-eligibility-v2"
MIN_CANONICAL_TEXT_LENGTH = 50
MAX_CANONICAL_TEXT_LENGTH = 1500
SUPERSEDED_V1_SELECTION_MANIFEST_SHA256 = (
    "a3ff8e783bbec0d231049536ddc532f816eb7f7756f4929b8eb065e0f2eed66f"
)


@dataclass(frozen=True)
class Phase11StructuralMetadata:
    """Structural metadata retained by the upstream statutory source."""

    structural_role: Literal["article", "article_part", "unresolved"]
    article_ordinal: int | None
    part_index: int | None
    parse_confidence: str


def phase11_unit_eligible(
    unit: CanonicalUnit, metadata: Phase11StructuralMetadata
) -> bool:
    """Apply the versioned, atomic article-sized Phase 11 selection policy."""

    return (
        unit.unit_type is UnitType.ARTICLE
        and metadata.structural_role == "article"
        and metadata.article_ordinal is not None
        and metadata.part_index is None
        and metadata.parse_confidence == "high"
        and MIN_CANONICAL_TEXT_LENGTH <= len(unit.text) <= MAX_CANONICAL_TEXT_LENGTH
        and unit.provenance.source_field == "text"
    )


class AnnotationRecord(ExtractionModel):
    canonical_unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    canonical_text: str
    source_provenance: SourceProvenance
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str
    smoke: bool = False
    strata: tuple[str, ...] = ()
    candidate_registry: CandidateRegistry
    human_annotations: SemanticProposal | None = None
    annotation_status: Literal[
        "unreviewed",
        "in_review",
        "reviewed",
        "independent_ai_review",
        "dual_ai_agreed",
        "dual_ai_disagreement",
        "human_adjudicated",
    ] = "unreviewed"
    annotation_provenance: Literal[
        "unreviewed",
        "independent_ai_review",
        "ai_adjudicated_after_independent_second_review",
        "dual_ai_agreed",
        "dual_ai_disagreement",
        "human_adjudicated",
    ] = "unreviewed"
    human_verified: bool = False


class AnnotationUpdate(ExtractionModel):
    """Private human update accepted by the DEV annotation helper."""

    human_annotations: SemanticProposal
    annotation_status: Literal["in_review", "reviewed"]
    human_verified: bool = False


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=64)
def _raw_structural_metadata_by_path(raw_path: str) -> tuple[Phase11StructuralMetadata, ...]:
    path = Path(raw_path)
    if not path.is_file():
        return ()
    schema = pq.read_schema(path)
    if "article_number" not in schema.names:
        return ()
    table: Any = pq.read_table(path, columns=["article_number"])
    metadata: list[Phase11StructuralMetadata] = []
    for value in table.column("article_number").to_pylist():
        parsed = parse_article_label(str(value or ""))
        if parsed.ordinal is None:
            role: Literal["article", "article_part", "unresolved"] = "unresolved"
        elif parsed.part is None:
            role = "article"
        else:
            role = "article_part"
        metadata.append(
            Phase11StructuralMetadata(
                structural_role=role,
                article_ordinal=parsed.ordinal,
                part_index=parsed.part,
                parse_confidence=parsed.article_parse_confidence.value,
            )
        )
    return tuple(metadata)


def _raw_structural_metadata(provenance: SourceProvenance) -> Phase11StructuralMetadata:
    raw_path = (
        Path("data/raw")
        / provenance.source_id
        / provenance.source_version
        / provenance.source_path
    )
    row_index = provenance.source_row - 1
    metadata = _raw_structural_metadata_by_path(raw_path.as_posix())
    if row_index < 0 or row_index >= len(metadata):
        return Phase11StructuralMetadata("unresolved", None, None, "unresolved")
    return metadata[row_index]


def _load_eligible_units() -> tuple[list[CanonicalUnit], str]:
    records = {record.source_id: record for record in load_registry()}
    units: list[CanonicalUnit] = []
    file_hashes: list[str] = []
    for path in sorted(CANONICAL_ROOT.glob("*/*/units.parquet")):
        table: Any = pq.read_table(path)
        file_hashes.append(f"{path.as_posix()}:{_file_hash(path)}")
        for raw in cast(list[dict[str, object]], table.to_pylist()):
            provenance = SourceProvenance(
                source_id=str(raw["source_id"]),
                source_version=str(raw["source_version"]),
                source_path=str(raw["source_path"]),
                source_row=int(cast(int | str, raw["source_row"])),
                source_field=str(raw["source_field"]),
                split=str(raw.get("split", "")),
            )
            unit = CanonicalUnit(
                unit_id=str(raw["unit_id"]),
                document_id=str(raw["document_id"]),
                unit_type=UnitType(str(raw["unit_type"])),
                text=str(raw["text"]),
                provenance=provenance,
                ordinal=(
                    int(cast(int | str, raw["ordinal"])) if raw.get("ordinal") is not None else None
                ),
            )
            source = records.get(unit.provenance.source_id)
            if source is not None and eligible_regulatory_unit(source, unit.unit_type):
                metadata = _raw_structural_metadata(unit.provenance)
                if phase11_unit_eligible(unit, metadata):
                    units.append(unit)
    if not units:
        raise ValueError("no eligible canonical statutory units are available")
    corpus_hash = hashlib.sha256("\n".join(sorted(file_hashes)).encode("utf-8")).hexdigest()
    return units, corpus_hash


def _weak_strata(text: str) -> tuple[str, ...]:
    cues = {
        "normative": r"يجب|يلتزم|يحظر|لا يجوز|يجوز|يتعين",
        "exception": r"إلا|باستثناء|ما لم|على الرغم",
        "penalty": r"غرامة|يعاقب|جزاء|عقوبة",
        "temporal": r"\d|يوم|شهر|سنة|تاريخ",
        "money_percentage": r"ريال|SAR|٪|%",
        "reference": r"المادة|نظام|لائحة",
    }
    found = tuple(sorted(name for name, pattern in cues.items() if re.search(pattern, text)))
    semantic_cues = tuple(name for name in found if name != "reference")
    if not semantic_cues:
        return ("low_signal",)
    return semantic_cues + (("reference",) if "reference" in found else ())


def _rank(unit: CanonicalUnit, corpus_hash: str) -> str:
    return hashlib.sha256(
        f"{_SEED}:{PHASE11_SELECTION_VERSION}:{PHASE11_ELIGIBILITY_POLICY_VERSION}:"
        f"{corpus_hash}:{unit.unit_id}".encode()
    ).hexdigest()


def _selection_rows(units: list[CanonicalUnit], corpus_hash: str) -> list[dict[str, object]]:
    unique_texts: set[str] = set()
    candidates: list[tuple[str, CanonicalUnit, tuple[str, ...]]] = []
    for unit in units:
        text_hash = hashlib.sha256(unit.text.encode("utf-8")).hexdigest()
        if text_hash in unique_texts:
            continue
        unique_texts.add(text_hash)
        candidates.append((_rank(unit, corpus_hash), unit, _weak_strata(unit.text)))
    candidates.sort(key=lambda item: item[0])
    holdout: list[tuple[CanonicalUnit, tuple[str, ...]]] = []
    holdout_documents: set[str] = set()
    for _, unit, strata in candidates:
        if strata == ("low_signal",) and unit.document_id not in holdout_documents:
            holdout.append((unit, strata))
            holdout_documents.add(unit.document_id)
            break
    for _, unit, strata in candidates:
        if len(holdout) == 40:
            break
        if unit.document_id in holdout_documents:
            continue
        holdout.append((unit, strata))
        holdout_documents.add(unit.document_id)
    if len(holdout) != 40:
        raise ValueError("unable to create the protected HOLDOUT selection")
    selected_ids = {unit.unit_id for unit, _ in holdout}
    dev: list[tuple[CanonicalUnit, tuple[str, ...]]] = []
    dev_selected_ids: set[str] = set()
    for _, unit, strata in candidates:
        if (
            strata == ("low_signal",)
            and unit.unit_id not in selected_ids
            and unit.document_id not in holdout_documents
        ):
            dev.append((unit, strata))
            dev_selected_ids.add(unit.unit_id)
            break
    for _, unit, strata in candidates:
        if len(dev) == 80:
            break
        if (
            unit.unit_id in selected_ids
            or unit.unit_id in dev_selected_ids
            or unit.document_id in holdout_documents
        ):
            continue
        dev.append((unit, strata))
        dev_selected_ids.add(unit.unit_id)
    if len(dev) != 80:
        raise ValueError("unable to create the document-disjoint DEV selection")
    rows: list[dict[str, object]] = []
    for split, selected in (("holdout", holdout), ("dev", dev)):
        for unit, strata in selected:
            rows.append(
                {
                    "canonical_unit_id": unit.unit_id,
                    "document_id": unit.document_id,
                    "source_id": unit.provenance.source_id,
                    "source_fingerprint": corpus_hash,
                    "split": split,
                    "smoke": split == "dev" and sum(item["split"] == "dev" for item in rows) < 10,
                    "strata": list(strata),
                }
            )
    return rows


def _selection_fingerprint(rows: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        {
            "selection_version": PHASE11_SELECTION_VERSION,
            "eligibility_policy_version": PHASE11_ELIGIBILITY_POLICY_VERSION,
            "rows": rows,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_annotation_record(
    record: AnnotationRecord,
    selection_unit_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    if record.canonical_unit_id not in selection_unit_ids:
        errors.append("provenance unit is not in the selection manifest")
    if record.annotation_status not in {
        "unreviewed",
        "in_review",
        "reviewed",
        "independent_ai_review",
        "dual_ai_agreed",
        "dual_ai_disagreement",
        "human_adjudicated",
    }:
        errors.append("annotation status is invalid")
    if (
        record.annotation_provenance
        in {
            "independent_ai_review",
            "ai_adjudicated_after_independent_second_review",
            "dual_ai_agreed",
            "dual_ai_disagreement",
        }
        and record.human_verified
    ):
        errors.append("AI-reviewed records cannot be human_verified")
    if record.human_verified and record.annotation_status not in {
        "reviewed",
        "human_adjudicated",
    }:
        errors.append("human_verified records must be reviewed")
    if record.candidate_registry.canonical_text != record.canonical_text:
        errors.append("candidate registry text differs from canonical text")
    for candidate in record.candidate_registry.candidates:
        if (
            record.canonical_text[candidate.span.start_char : candidate.span.end_char]
            != candidate.span.text
        ):
            errors.append(f"candidate span is not exact: {candidate.candidate_id}")
    candidate_ids = {candidate.candidate_id for candidate in record.candidate_registry.candidates}
    if record.human_annotations is None:
        return errors
    def validate_span_collection(spans: tuple[Any, ...], label: str) -> None:
        seen: set[tuple[int, int, str]] = set()
        for span in spans:
            try:
                resolved = resolve_exact_span(
                    record.canonical_text,
                    span.text,
                    occurrence=span.occurrence,
                    canonical_unit_id=record.canonical_unit_id,
                    document_id=record.document_id,
                )
            except ValueError as error:
                errors.append(f"semantic span is not exact: {error}")
                continue
            # The resolved codepoint offsets encode the selected occurrence.
            identity = (resolved.start_char, resolved.end_char, resolved.text)
            if identity in seen:
                errors.append(f"duplicate {label} is not allowed")
            seen.add(identity)

    proposal = record.human_annotations
    validate_span_collection(proposal.regulated_entities, "regulated entity")
    validate_span_collection(proposal.exceptions, "top-level exception")
    validate_span_collection(proposal.penalties, "top-level penalty")
    seen_rules: list[Any] = []
    for index, rule in enumerate(proposal.rules):
        if any(rule == previous for previous in seen_rules):
            errors.append(f"duplicate normative rule at index {index} is not allowed")
        seen_rules.append(rule)
        if rule.actor is not None:
            validate_span_collection((rule.actor,), f"rule {index} actor")
        validate_span_collection((rule.action,), f"rule {index} action")
        validate_span_collection(rule.conditions, f"rule {index} condition")
        validate_span_collection(rule.exceptions, f"rule {index} exception")

    def validate_candidate_collection(
        refs: tuple[str, ...], expected_prefix: str, label: str
    ) -> None:
        if len(set(refs)) != len(refs):
            errors.append(f"duplicate {label} candidate reference is not allowed")
        for candidate_id in refs:
            if not candidate_id.startswith(expected_prefix):
                errors.append(f"candidate reference has the wrong type: {candidate_id}")
            elif candidate_id not in candidate_ids:
                errors.append(f"unknown candidate reference: {candidate_id}")

    for refs, expected, label in (
        (proposal.deadline_refs, "T", "deadline"),
        (proposal.effective_date_refs, "T", "effective-date"),
        (proposal.monetary_threshold_refs, "M", "monetary-threshold"),
        (proposal.percentage_threshold_refs, "P", "percentage-threshold"),
    ):
        validate_candidate_collection(refs, expected, label)
    for index, rule in enumerate(proposal.rules):
        for refs, expected, label in (
            (rule.deadline_refs, "T", f"rule {index} deadline"),
            (rule.effective_date_refs, "T", f"rule {index} effective-date"),
            (rule.monetary_threshold_refs, "M", f"rule {index} monetary-threshold"),
            (rule.percentage_threshold_refs, "P", f"rule {index} percentage-threshold"),
        ):
            validate_candidate_collection(refs, expected, label)
    return errors


def is_human_gold(record: AnnotationRecord) -> bool:
    return (
        record.annotation_status in {"reviewed", "human_adjudicated"}
        and record.annotation_provenance
        not in {
            "independent_ai_review",
            "ai_adjudicated_after_independent_second_review",
            "dual_ai_agreed",
            "dual_ai_disagreement",
        }
        and record.human_verified
    )


def prepare_annotation_pack(
    *,
    private_root: Path = ANNOTATION_ROOT,
    manifest_path: Path = SELECTION_MANIFEST_PATH,
) -> dict[str, object]:
    units, corpus_hash = _load_eligible_units()
    rows = _selection_rows(units, corpus_hash)
    by_id = {unit.unit_id: unit for unit in units}
    selection_fingerprint = _selection_fingerprint(rows)
    records: list[AnnotationRecord] = []
    private_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        unit = by_id[str(row["canonical_unit_id"])]
        record = AnnotationRecord(
            canonical_unit_id=unit.unit_id,
            document_id=unit.document_id,
            canonical_text=unit.text,
            source_provenance=unit.provenance,
            source_fingerprint=corpus_hash,
            split=str(row["split"]),
            smoke=bool(row["smoke"]),
            strata=tuple(cast(list[str], row["strata"])),
            candidate_registry=build_candidate_registry(
                unit.text,
                canonical_unit_id=unit.unit_id,
                document_id=unit.document_id,
            ),
        )
        records.append(record)
        private_path = private_root / f"{hashlib.sha256(unit.unit_id.encode()).hexdigest()}.json"
        write_private_json(private_path, record.model_dump(mode="json"))
    dev_ids = {str(row["canonical_unit_id"]) for row in rows if row["split"] == "dev"}
    holdout_ids = {str(row["canonical_unit_id"]) for row in rows if row["split"] == "holdout"}
    dev_documents = {str(row["document_id"]) for row in rows if row["split"] == "dev"}
    holdout_documents = {str(row["document_id"]) for row in rows if row["split"] == "holdout"}
    lengths = sorted(len(record.canonical_text) for record in records)
    candidate_counts: dict[str, int] = {}
    candidate_records: dict[str, int] = {}
    for record in records:
        seen: set[str] = set()
        for candidate in record.candidate_registry.candidates:
            kind = candidate.candidate_type.value
            candidate_counts[kind] = candidate_counts.get(kind, 0) + 1
            seen.add(kind)
        for kind in seen:
            candidate_records[kind] = candidate_records.get(kind, 0) + 1
    manifest = {
        "schema_version": 2,
        "artifact_type": "phase11_annotation_selection",
        "supersedes": {
            "selection_version": "phase11-selection-v1",
            "manifest_sha256": SUPERSEDED_V1_SELECTION_MANIFEST_SHA256,
            "status": "superseded",
        },
        "selection_version": PHASE11_SELECTION_VERSION,
        "eligibility_policy_version": PHASE11_ELIGIBILITY_POLICY_VERSION,
        "candidate_registry_version": CANDIDATE_REGISTRY_VERSION,
        "selection_fingerprint": selection_fingerprint,
        "corpus_fingerprint": corpus_hash,
        "source_universe": "governed_primary_statutory_corpus",
        "eligible_corpus_counts": {
            "documents": len({unit.document_id for unit in units}),
            "units": len(units),
        },
        "selection_counts": {
            "total": len(rows),
            "dev": len(dev_ids),
            "holdout": len(holdout_ids),
            "smoke": sum(bool(row["smoke"]) for row in rows),
        },
        "document_disjoint": dev_documents.isdisjoint(holdout_documents),
        "holdout_sealed": True,
        "annotation_state_counts": {"unreviewed": len(rows), "human_verified": 0},
        "text_length_distribution": {
            "min": min(lengths),
            "p50": lengths[(len(lengths) - 1) // 2],
            "p95": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95) - 1)],
            "max": max(lengths),
            "count_over_1000": sum(length > 1000 for length in lengths),
            "count_over_1500": sum(length > 1500 for length in lengths),
        },
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "records_containing_candidate_type": dict(sorted(candidate_records.items())),
        "rows": rows,
    }
    write_text_free_json(manifest_path, manifest)
    return {**manifest, "records": records}
