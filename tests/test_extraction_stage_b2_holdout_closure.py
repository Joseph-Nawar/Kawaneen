from __future__ import annotations

from kawaneen.extraction.stage_b2_holdout_evaluation import (
    evaluate_clean_stage_b2_holdout,
)


def test_frozen_holdout_evaluation_reconciles_completion_and_safety() -> None:
    report = evaluate_clean_stage_b2_holdout()

    assert report["record_count"] == 40
    assert report["completed_record_count"] == 39
    assert report["failed_record_count"] == 1
    assert report["failure_diagnostics"][0]["record_id"] == ("f52c0c03-a024-513c-ac04-f79b16e9a234")
    safety = report["safety_structural_metrics"]
    assert safety["FinalSchemaValidityRate"]["completed_outputs"]["rate"] == 1.0
    assert safety["UnsupportedSpanAcceptanceRate"]["rate"] == 0.0
    assert safety["InvalidCandidateReferenceAcceptanceRate"]["rate"] == 0.0
