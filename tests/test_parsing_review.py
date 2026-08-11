import json
from pathlib import Path

import pytest

from kawaneen.parsing.review import (
    ExternalReviewEvidenceError,
    validate_external_gold,
    validate_external_statutory_reconciliation,
)


def test_external_gold_is_blocked_when_not_supplied(tmp_path: Path) -> None:
    result = validate_external_gold(tmp_path / "missing.jsonl")
    assert result["status"] == "blocked_missing_external_evidence"


def test_external_gold_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    benchmark = tmp_path / "benchmark.json"
    selection.write_text(
        json.dumps({"selection": [{"id": "page-1", "source_pdf": "source.pdf"}]}),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps({"source_pdfs": [{"filename": "source.pdf", "sha256": "a" * 64}]}),
        encoding="utf-8",
    )
    external = tmp_path / "gold.jsonl"
    external.write_text(
        json.dumps(
            {
                "page_identity": {
                    "page_id": "page-1",
                    "source_pdf_filename": "source.pdf",
                    "source_pdf_sha256": "b" * 64,
                },
                "candidate_regions": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExternalReviewEvidenceError, match="hash mismatch"):
        validate_external_gold(external, selection, benchmark)


def test_external_gold_validates_identity_without_marking_human_review(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    benchmark = tmp_path / "benchmark.json"
    selection.write_text(
        json.dumps({"selection": [{"id": "page-1", "source_pdf": "source.pdf"}]}),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps({"source_pdfs": [{"filename": "source.pdf", "sha256": "a" * 64}]}),
        encoding="utf-8",
    )
    external = tmp_path / "gold.jsonl"
    regions = [
        {
            "human_verified": False,
            "adjudicated_text": "",
            "adjudicated_region_type": "other_structural",
        }
        for _ in range(102)
    ]
    external.write_text(
        json.dumps(
            {
                "page_identity": {
                    "page_id": "page-1",
                    "source_pdf_filename": "source.pdf",
                    "source_pdf_sha256": "a" * 64,
                },
                "candidate_regions": regions,
            }
        ),
        encoding="utf-8",
    )
    result = validate_external_gold(external, selection, benchmark)
    assert result["status"] == "externally_source_verified_pending_human_review"
    assert result["human_verified"] is False
    assert result["region_count"] == 102


def test_external_gold_accepts_adjudication_regions_and_preserves_ai_status(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    benchmark = tmp_path / "benchmark.json"
    selection.write_text(
        json.dumps({"selection": [{"id": "page-1", "source_pdf": "source.pdf"}]}),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps({"source_pdfs": [{"filename": "source.pdf", "sha256": "a" * 64}]}),
        encoding="utf-8",
    )
    external = tmp_path / "gold.jsonl"
    regions = [
        {
            "human_verified": False,
            "adjudicated_text": "",
            "adjudicated_region_type": "other_structural",
        }
        for _ in range(102)
    ]
    external.write_text(
        json.dumps(
            {
                "page_identity": {
                    "page_id": "page-1",
                    "source_pdf_filename": "source.pdf",
                    "source_pdf_sha256": "a" * 64,
                },
                "regions": regions,
            }
        ),
        encoding="utf-8",
    )
    result = validate_external_gold(external, selection, benchmark)
    assert result["status"] == "externally_source_verified_pending_human_review"
    assert result["reviewer_type"] == "independent_ai_visual_review"
    assert result["human_verified"] is False


def test_external_gold_records_metric_scope_correction(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    benchmark = tmp_path / "benchmark.json"
    selection.write_text(
        json.dumps({"selection": [{"id": "page-1", "source_pdf": "source.pdf"}]}),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps(
            {
                "source_pdfs": [{"filename": "source.pdf", "sha256": "a" * 64}],
                "metric_scopes": {"legal_structure": {"page_ids": ["page-1"]}},
            }
        ),
        encoding="utf-8",
    )
    external = tmp_path / "gold.jsonl"
    external.write_text(
        json.dumps(
            {
                "page_identity": {
                    "page_id": "page-1",
                    "source_pdf_filename": "source.pdf",
                    "source_pdf_sha256": "a" * 64,
                    "metric_scope": "legal_ocr",
                },
                "regions": [
                    {
                        "human_verified": False,
                        "adjudicated_text": "",
                        "adjudicated_region_type": "other_structural",
                    }
                    for _ in range(102)
                ],
            }
        ),
        encoding="utf-8",
    )
    result = validate_external_gold(external, selection, benchmark)
    assert result["scope_corrections"][0] == {
        "page_id": "page-1",
        "supplied_scope": "legal_ocr",
        "canonical_scope": "legal_structure",
    }


def test_external_statutory_reconciliation_is_ai_evidence_and_never_v1_eligible(
    tmp_path: Path,
) -> None:
    external = tmp_path / "statutory.jsonl"
    external.write_text(
        json.dumps(
            {
                "record_type": "phase3_authoritative_statutory_reconciliation",
                "reviewer": {
                    "review_type": "independent_ai_source_review",
                    "human_verified": False,
                },
                "law": {"dataset_law_name": "Law"},
                "sample_review": {
                    "requested_samples": 5,
                    "results": [{"target_present_in_exported_sample": True}] * 5,
                },
                "reconciliation_decision": {
                    "review_process_status": "completed_with_documented_partial_result",
                    "corpus_status": "present_partial_reviewed_not_eligible",
                    "eligible_for_kawaneen_v1_statutory_corpus": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_external_statutory_reconciliation(external, expected_laws=("Law",))

    assert result["status"] == "completed_partial_not_eligible"
    assert result["reviewer_type"] == "independent_ai_source_review"
    assert result["human_verified"] is False
    assert result["eligible_law_count"] == 0


def test_external_statutory_reconciliation_blocks_missing_evidence(tmp_path: Path) -> None:
    result = validate_external_statutory_reconciliation(
        tmp_path / "missing.jsonl", expected_laws=()
    )

    assert result["status"] == "blocked_missing_external_statutory_evidence"
