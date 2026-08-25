"""Text-free readiness summaries for the private Phase 11A annotation pack."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import cast

from kawaneen.extraction.annotation import AnnotationRecord
from kawaneen.extraction.artifacts import write_text_free_json
from kawaneen.extraction.deterministic import run_deterministic

READINESS_MANIFEST_PATH = Path("data/manifests/extraction/phase11_readiness.json")
READINESS_REPORT_PATH = Path("data/evaluation/phase11_readiness.json")


def build_readiness_report(pack: dict[str, object]) -> dict[str, object]:
    records = cast(list[AnnotationRecord], pack["records"])
    candidate_counts = Counter(
        candidate.candidate_type.value
        for record in records
        for candidate in record.candidate_registry.candidates
    )
    units_with_type = {
        candidate_type: sum(
            any(
                candidate.candidate_type.value == candidate_type
                for candidate in record.candidate_registry.candidates
            )
            for record in records
        )
        for candidate_type in ("temporal", "monetary", "percentage", "article", "regulation")
    }
    strata_counts = Counter(stratum for record in records for stratum in record.strata)
    status_counts = Counter(record.annotation_status for record in records)
    smoke = [record for record in records if record.smoke]
    smoke_success = all(
        run_deterministic(
            record.canonical_text,
            canonical_unit_id=record.canonical_unit_id,
            document_id=record.document_id,
            source_provenance=record.source_provenance,
        ).schema_version
        == "phase11-extraction-v1"
        for record in smoke
    )
    selection_counts = cast(dict[str, int], pack["selection_counts"])
    return {
        "schema_version": 1,
        "artifact_type": "phase11_readiness",
        "eligible_corpus_counts": pack.get("eligible_corpus_counts", {}),
        "selection_counts": selection_counts,
        "selection_version": pack.get("selection_version"),
        "eligibility_policy_version": pack.get("eligibility_policy_version"),
        "candidate_registry_version": pack.get("candidate_registry_version"),
        "document_disjoint": pack["document_disjoint"],
        "holdout_sealed": pack["holdout_sealed"],
        "sampling_strata_distribution": dict(sorted(strata_counts.items())),
        "deterministic_candidate_counts": dict(sorted(candidate_counts.items())),
        "selected_units_containing_candidate_type": dict(sorted(units_with_type.items())),
        "selected_units_containing_candidate_type_percentage": {
            key: round(value / selection_counts["total"] * 100, 4)
            for key, value in units_with_type.items()
        },
        "text_length_distribution": pack.get("text_length_distribution", {}),
        "source_governance": {
            "source_universe": "governed_primary_statutory_corpus",
            "regulatory_only": True,
            "private_source_text": True,
            "permission_gate_authoritative": True,
        },
        "annotation_status_counts": {
            **dict(sorted(status_counts.items())),
            "human_verified": sum(record.human_verified for record in records),
        },
        "tracked_source_text_leakage": 0,
        "deterministic_extractor_smoke_success": smoke_success,
        "hybrid_model_calls": 0,
        "holdout_evaluation_performed": False,
        "semantic_extraction_performance_reported": False,
        "safety_targets": {
            "FinalSchemaValidityRate": 1.0,
            "UnsupportedSpanAcceptanceRate": 0.0,
            "InvalidCandidateReferenceAcceptanceRate": 0.0,
            "ProvenanceCompletenessRate": 1.0,
        },
    }


def write_readiness_artifacts(
    pack: dict[str, object],
    *,
    manifest_path: Path = READINESS_MANIFEST_PATH,
    report_path: Path = READINESS_REPORT_PATH,
) -> dict[str, object]:
    report = build_readiness_report(pack)
    write_text_free_json(manifest_path, report)
    write_text_free_json(report_path, report)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {**report, "readiness_manifest_sha256": digest}
