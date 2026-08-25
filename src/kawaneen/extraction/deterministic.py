"""Deterministic Phase 11A extraction over one canonical unit."""

from __future__ import annotations

import hashlib

from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.candidates import build_candidate_registry
from kawaneen.extraction.contracts import (
    CandidateType,
    ExtractionResult,
    FieldProvenance,
    ProvenanceOrigin,
    ValidationMetadata,
)


def run_deterministic(
    canonical_text: str,
    *,
    canonical_unit_id: str,
    document_id: str,
    source_provenance: SourceProvenance,
    issuing_authority: str | None = None,
) -> ExtractionResult:
    registry = build_candidate_registry(
        canonical_text,
        canonical_unit_id=canonical_unit_id,
        document_id=document_id,
    )
    provenance = tuple(
        FieldProvenance(field_name=field_name, origin=ProvenanceOrigin.DETERMINISTIC)
        for field_name in (
            "regulated_entities",
            "rules",
            "deadlines",
            "effective_dates",
            "penalties",
            "monetary_thresholds",
            "percentage_thresholds",
            "exceptions",
            "referenced_articles",
            "referenced_regulations",
        )
    )
    return ExtractionResult(
        schema_version="phase11-extraction-v1",
        extractor_version="deterministic-v1",
        configuration="deterministic-v1",
        jurisdiction="SA",
        source_provenance=source_provenance,
        source_fingerprint=hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
        issuing_authority=issuing_authority,
        candidate_registry=registry,
        referenced_articles=tuple(
            item for item in registry.candidates if item.candidate_type is CandidateType.ARTICLE
        ),
        referenced_regulations=tuple(
            item for item in registry.candidates if item.candidate_type is CandidateType.REGULATION
        ),
        validation_metadata=ValidationMetadata(),
        field_provenance=(
            *provenance,
            FieldProvenance(field_name="issuing_authority", origin=ProvenanceOrigin.METADATA),
            FieldProvenance(
                field_name="deterministic_candidates",
                origin=ProvenanceOrigin.DETERMINISTIC,
                source_ids=tuple(item.candidate_id for item in registry.candidates),
            ),
        ),
    )
