"""Validation helpers for independently supplied parser-review evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast


class ExternalReviewEvidenceError(ValueError):
    """Raised when external gold evidence does not match the frozen benchmark."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_external_gold(
    external_path: Path,
    selection_manifest_path: Path = Path(
        "artifacts/private/parsing_benchmark/selection_manifest.json"
    ),
    benchmark_manifest_path: Path = Path("data/manifests/parsing_benchmark.json"),
) -> dict[str, Any]:
    """Validate external page identity and source hashes before integration.

    This function does not mark any region human verified. External AI visual review
    remains a separate provenance state and is rejected if the frozen page identity
    or source-PDF hash differs.
    """

    if not external_path.is_file():
        return {
            "status": "blocked_missing_external_evidence",
            "path": external_path.as_posix(),
            "reason": "independent adjudication JSONL was not supplied",
        }
    frozen = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    source_hashes = {item["filename"]: item["sha256"] for item in benchmark.get("source_pdfs", [])}
    scope_by_page = {
        page_id: scope
        for scope, payload in benchmark.get("metric_scopes", {}).items()
        for page_id in payload.get("page_ids", [])
    }
    expected = {
        item["id"]: {
            "source_pdf_filename": item["source_pdf"],
            "source_pdf_sha256": source_hashes.get(item["source_pdf"]),
        }
        for item in frozen.get("selection", [])
    }
    records: list[dict[str, Any]] = [
        cast(dict[str, Any], json.loads(line))
        for line in external_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != len(expected):
        raise ExternalReviewEvidenceError(
            f"expected {len(expected)} page records, received {len(records)}"
        )
    seen: set[str] = set()
    region_count = 0
    scope_corrections: list[dict[str, str]] = []
    for record in records:
        identity = record.get("page_identity", {})
        page_id = identity.get("page_id")
        if page_id not in expected or page_id in seen:
            raise ExternalReviewEvidenceError(f"unknown or duplicate page ID: {page_id}")
        frozen_page = expected[page_id]
        if identity.get("source_pdf_filename") != frozen_page.get("source_pdf_filename"):
            raise ExternalReviewEvidenceError(f"source PDF mismatch for {page_id}")
        if identity.get("source_pdf_sha256") != frozen_page.get("source_pdf_sha256"):
            raise ExternalReviewEvidenceError(f"source PDF hash mismatch for {page_id}")
        expected_scope = scope_by_page.get(page_id)
        if expected_scope is not None and identity.get("metric_scope") != expected_scope:
            scope_corrections.append(
                {
                    "page_id": page_id,
                    "supplied_scope": str(identity.get("metric_scope")),
                    "canonical_scope": expected_scope,
                }
            )
        if identity.get("human_verified") is True:
            raise ExternalReviewEvidenceError("external AI evidence cannot set human_verified")
        is_legacy_candidate_bundle = "regions" not in record and "candidate_regions" in record
        regions_value = record.get("regions", record.get("candidate_regions", []))
        if not isinstance(regions_value, list):
            raise ExternalReviewEvidenceError(f"regions must be a list for {page_id}")
        region_items = cast(list[Any], regions_value)
        if not all(isinstance(item, dict) for item in region_items):
            raise ExternalReviewEvidenceError(f"regions must be a list for {page_id}")
        regions = [cast(dict[str, Any], item) for item in region_items]
        for region in regions:
            if region.get("human_verified") is True:
                raise ExternalReviewEvidenceError("external AI evidence cannot set human_verified")
            if expected_scope is not None and region.get("metric_scope") != expected_scope:
                scope_corrections.append(
                    {
                        "page_id": page_id,
                        "region_id": str(region.get("region_id")),
                        "supplied_scope": str(region.get("metric_scope")),
                        "canonical_scope": expected_scope,
                    }
                )
            if not is_legacy_candidate_bundle and (
                "adjudicated_text" not in region or "adjudicated_region_type" not in region
            ):
                raise ExternalReviewEvidenceError(f"incomplete adjudicated region for {page_id}")
            region_count += 1
        seen.add(page_id)
    if region_count != 102:
        raise ExternalReviewEvidenceError(f"expected 102 regions, received {region_count}")
    return {
        "status": "externally_source_verified_pending_human_review",
        "reviewer_type": "independent_ai_visual_review",
        "page_count": len(records),
        "region_count": region_count,
        "source_pdf_hashes_validated": True,
        "human_verified": False,
        "scope_corrections": scope_corrections,
        "source_artifact_sha256": _sha256(external_path),
    }


def validate_external_statutory_reconciliation(
    external_path: Path, *, expected_laws: tuple[str, ...]
) -> dict[str, Any]:
    """Validate the closed, negative statutory reconciliation decision.

    The external artefact is independent AI source review only.  This validator is
    intentionally incapable of accepting human verification or a positive v1
    statutory eligibility result.
    """

    if not external_path.is_file():
        return {
            "status": "blocked_missing_external_statutory_evidence",
            "path": external_path.as_posix(),
        }
    records = [
        cast(dict[str, Any], json.loads(line))
        for line in external_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_law: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("record_type") != "phase3_authoritative_statutory_reconciliation":
            raise ExternalReviewEvidenceError("unexpected statutory reconciliation record type")
        reviewer = cast(dict[str, Any], record.get("reviewer", {}))
        if reviewer.get("review_type") != "independent_ai_source_review":
            raise ExternalReviewEvidenceError(
                "statutory reconciliation must remain independent AI review"
            )
        if reviewer.get("human_verified") is not False:
            raise ExternalReviewEvidenceError(
                "external statutory reconciliation cannot claim human review"
            )
        law = cast(dict[str, Any], record.get("law", {})).get("dataset_law_name")
        if not isinstance(law, str) or law in by_law:
            raise ExternalReviewEvidenceError(
                "statutory reconciliation law identity is missing or duplicated"
            )
        decision = cast(dict[str, Any], record.get("reconciliation_decision", {}))
        if decision.get("review_process_status") != "completed_with_documented_partial_result":
            raise ExternalReviewEvidenceError(
                "statutory review process is not a completed partial result"
            )
        if decision.get("corpus_status") != "present_partial_reviewed_not_eligible":
            raise ExternalReviewEvidenceError("statutory corpus status is not fail-closed")
        if decision.get("eligible_for_kawaneen_v1_statutory_corpus") is not False:
            raise ExternalReviewEvidenceError(
                "external statutory review cannot grant v1 eligibility"
            )
        sample_review = cast(dict[str, Any], record.get("sample_review", {}))
        if sample_review.get("requested_samples") != 5:
            raise ExternalReviewEvidenceError("each law must retain five requested review samples")
        results = cast(list[Any] | None, sample_review.get("results"))
        if not isinstance(results, list) or len(results) != 5:
            raise ExternalReviewEvidenceError("each law must retain five sample results")
        by_law[law] = record
    if set(by_law) != set(expected_laws):
        raise ExternalReviewEvidenceError(
            "statutory reconciliation laws do not match the frozen candidate set"
        )
    return {
        "status": "completed_partial_not_eligible",
        "reviewer_type": "independent_ai_source_review",
        "human_verified": False,
        "law_count": len(by_law),
        "eligible_law_count": 0,
        "source_artifact_sha256": _sha256(external_path),
    }
