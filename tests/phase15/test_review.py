from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawaneen.phase15.contracts import (
    ErrorCategory,
    ReviewCase,
    ReviewDecision,
    ReviewOutcome,
)
from kawaneen.phase15.review import (
    ReviewStore,
    aggregate_review_decisions,
    ai_human_agreement,
    build_review_manifest,
    default_review_paths,
    prepare_review_packet,
)


def _cases() -> tuple[ReviewCase, ...]:
    return tuple(
        ReviewCase(
            case_id=f"case-{i}",
            language="ar" if i % 2 else "en",
            pipeline_stage="retrieval" if i % 3 else "generation",
            legal_category="civil" if i % 2 else "labor",
            answerability="answerable" if i % 4 else "unanswerable",
            severity="high" if i % 5 == 0 else "medium",
            query_text=f"private query {i}",
            evidence_text=f"private evidence {i}",
        )
        for i in range(120)
    )


def test_packet_is_exactly_120_dev_cases_and_manifest_is_text_free(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    manifest_path = tmp_path / "manifest.json"
    prepare_review_packet(_cases(), packet_path, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["case_count"] == 120
    assert "private query" not in manifest_path.read_text()
    packet = json.loads(packet_path.read_text())
    assert len(packet["cases"]) == 120
    assert all(not case["holdout"] for case in packet["cases"])


def test_manifest_counts_unavailable_ai_attempts_without_assigning_categories() -> None:
    cases = tuple(
        case.model_copy(update={"ai_preclassification_attempted": True}) for case in _cases()
    )
    manifest = build_review_manifest(cases)
    assert manifest["ai_preclassification_attempted"] == 120
    assert manifest["ai_preclassification_successful"] == 0
    assert manifest["ai_preclassification_unavailable"] == 120
    assert all(case.ai_suggestion is None for case in cases)


def test_manifest_separates_successful_and_unavailable_ai_preclassification() -> None:
    cases = tuple(
        case.model_copy(
            update={
                "ai_preclassification_attempted": True,
                "ai_suggestion": ErrorCategory.OCR_FAILURE if index < 90 else None,
            }
        )
        for index, case in enumerate(_cases())
    )
    manifest = build_review_manifest(cases)
    assert manifest["ai_preclassification_attempted"] == 120
    assert manifest["ai_preclassification_successful"] == 90
    assert manifest["ai_preclassification_unavailable"] == 30


def test_atomic_review_progress_resumes_and_deduplicates(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    prepare_review_packet(_cases(), packet_path, tmp_path / "manifest.json")
    store = ReviewStore(packet_path, tmp_path / "progress.json")
    decision = ReviewDecision(
        case_id="case-1",
        outcome=ReviewOutcome.CONFIRMED_FAILURE,
        primary=ErrorCategory.OCR_FAILURE,
    )
    store.save_decision(decision)
    store.save_decision(decision.model_copy(update={"confidence": 5}))
    assert store.reviewed_count() == 1
    assert ReviewStore(packet_path, tmp_path / "progress.json").reviewed_count() == 1


def test_finalize_hard_fails_before_100_unique_decisions(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    prepare_review_packet(_cases(), packet_path, tmp_path / "manifest.json")
    store = ReviewStore(packet_path, tmp_path / "progress.json")
    with pytest.raises(RuntimeError, match="100"):
        store.require_finalize_ready()


def test_review_store_status_and_packet_guards(tmp_path: Path) -> None:
    cases = _cases()
    packet_path = tmp_path / "packet.json"
    prepare_review_packet(cases, packet_path, tmp_path / "manifest.json")
    store = ReviewStore(packet_path, tmp_path / "progress.json")
    assert len(store.cases()) == 120
    assert store.next_unreviewed() is not None
    assert store.decision_for("missing") is None
    status = store.status()
    assert status["progress"] == "0 / 120"
    assert status["finalize_ready"] is False
    assert default_review_paths(tmp_path)[0].as_posix().endswith("review/review_packet.json")
    with pytest.raises(ValueError, match="unknown immutable"):
        store.save_decision(
            ReviewDecision(
                case_id="missing",
                outcome=ReviewOutcome.CONFIRMED_FAILURE,
                primary=ErrorCategory.OCR_FAILURE,
            )
        )

    assert build_review_manifest(cases)["holdout_case_count"] == 0
    with pytest.raises(ValueError, match="exactly 120"):
        build_review_manifest(cases[:1])


def test_review_outcome_aggregation_excludes_non_failures_from_taxonomy() -> None:
    decisions = (
        ReviewDecision(
            case_id="case-1",
            outcome=ReviewOutcome.CONFIRMED_FAILURE,
            primary=ErrorCategory.OCR_FAILURE,
        ),
        ReviewDecision(
            case_id="case-2",
            outcome=ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE,
        ),
        ReviewDecision(case_id="case-3", outcome=ReviewOutcome.UNCERTAIN),
    )
    summary = aggregate_review_decisions(decisions)
    assert summary["human_reviewed_count"] == 3
    assert summary["review_outcome_distribution"] == {
        "BORDERLINE_NO_CONFIRMED_FAILURE": 1,
        "CONFIRMED_FAILURE": 1,
        "UNCERTAIN": 1,
    }
    assert summary["confirmed_failure_taxonomy"] == {"OCR failure": 1}
    assert summary["borderline_no_confirmed_failure_count"] == 1
    assert summary["uncertain_count"] == 1


def test_ai_agreement_excludes_unavailable_and_nonconfirmed_reviews() -> None:
    cases = (
        _cases()[0].model_copy(update={"ai_suggestion": ErrorCategory.OCR_FAILURE}),
        _cases()[1],
        _cases()[2].model_copy(update={"ai_suggestion": ErrorCategory.OCR_FAILURE}),
    )
    decisions = (
        ReviewDecision(
            case_id="case-0",
            outcome=ReviewOutcome.CONFIRMED_FAILURE,
            primary=ErrorCategory.OCR_FAILURE,
        ),
        ReviewDecision(
            case_id="case-1",
            outcome=ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE,
        ),
        ReviewDecision(case_id="case-2", outcome=ReviewOutcome.UNCERTAIN),
    )
    assert ai_human_agreement(cases, decisions) == {
        "eligible_count": 1,
        "agreement_count": 1,
        "agreement_rate": 1.0,
    }
