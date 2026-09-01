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
    aggregate_ai_classifications,
    aggregate_review_decisions,
    ai_human_agreement,
    build_human_audit_manifest,
    build_review_manifest,
    default_review_paths,
    prepare_review_packet,
    select_human_audit_cases,
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
    cases = _cases()
    prepare_review_packet(cases, packet_path, tmp_path / "manifest.json")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(build_human_audit_manifest(cases)), encoding="utf-8")
    store = ReviewStore(packet_path, tmp_path / "progress.json", audit_path)
    case_id = json.loads(audit_path.read_text())["case_ids"][0]
    decision = ReviewDecision(
        case_id=case_id,
        outcome=ReviewOutcome.CONFIRMED_FAILURE,
        primary=ErrorCategory.OCR_FAILURE,
    )
    store.save_decision(decision)
    store.save_decision(decision.model_copy(update={"confidence": 5}))
    assert store.reviewed_count() == 1
    assert ReviewStore(packet_path, tmp_path / "progress.json", audit_path).reviewed_count() == 1


def test_finalize_hard_fails_before_100_unique_decisions(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    cases = _cases()
    prepare_review_packet(cases, packet_path, tmp_path / "manifest.json")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(build_human_audit_manifest(cases)), encoding="utf-8")
    store = ReviewStore(packet_path, tmp_path / "progress.json", audit_path)
    with pytest.raises(RuntimeError, match="30"):
        store.require_finalize_ready()


def test_review_store_status_and_packet_guards(tmp_path: Path) -> None:
    cases = _cases()
    packet_path = tmp_path / "packet.json"
    prepare_review_packet(cases, packet_path, tmp_path / "manifest.json")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(build_human_audit_manifest(cases)), encoding="utf-8")
    store = ReviewStore(packet_path, tmp_path / "progress.json", audit_path)
    assert len(store.cases()) == 120
    assert store.next_unreviewed() is not None
    assert store.decision_for("missing") is None
    status = store.status()
    assert status["progress"] == "0 / 30"
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
    agreement = ai_human_agreement(cases, decisions)
    assert agreement["eligible_count"] == 1
    assert agreement["agreement_count"] == 1
    assert agreement["agreement_rate"] == 1.0
    assert agreement["cohens_kappa"] is None
    assert agreement["disagreement_counts"] == {}


def _audit_cases() -> tuple[ReviewCase, ...]:
    stages = ("retrieval", "normalization", "generation", "dialect", "reranking")
    languages = ("ar", "en", "ar-en", "ar-egyptian", "ar-gulf_saudi", "ar-levantine")
    return tuple(
        ReviewCase(
            case_id=f"audit-case-{i}",
            language=languages[i % len(languages)],
            pipeline_stage=stages[i % len(stages)],
            legal_category=f"category-{i % 8}",
            answerability="unanswerable" if i % 10 == 0 else "answerable",
            severity="high" if i % 7 == 0 else "medium",
            ai_suggestion=ErrorCategory.OCR_FAILURE if i % 4 else None,
            ai_preclassification_attempted=True,
        )
        for i in range(120)
    )


def test_human_audit_selection_is_deterministic_dev_only_and_exactly_30() -> None:
    cases = _audit_cases()
    first = select_human_audit_cases(cases)
    second = select_human_audit_cases(cases)
    assert tuple(case.case_id for case in first) == tuple(case.case_id for case in second)
    assert len(first) == len({case.case_id for case in first}) == 30
    assert all(not case.holdout for case in first)
    with pytest.raises(ValueError, match="HOLDOUT"):
        select_human_audit_cases((*cases[:-1], cases[-1].model_copy(update={"holdout": True})))


def test_human_audit_manifest_tracks_population_hash_and_distributions(tmp_path: Path) -> None:
    cases = _audit_cases()
    manifest = build_human_audit_manifest(cases)
    assert manifest["count"] == 30
    assert manifest["selection_seed"] == 20260826
    assert manifest["population_case_count"] == 120
    assert manifest["population_case_ids_sha256"] == build_review_manifest(cases)["case_ids_sha256"]
    assert sum(manifest["pipeline_stage_distribution"].values()) == 30
    assert manifest["holdout_case_count"] == 0
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert json.loads(path.read_text())["case_ids_sha256"] == manifest["case_ids_sha256"]


def test_audit_progress_is_30_and_non_audit_decisions_do_not_finalize(tmp_path: Path) -> None:
    cases = _audit_cases()
    packet_path = tmp_path / "packet.json"
    prepare_review_packet(cases, packet_path, tmp_path / "manifest.json")
    audit_manifest_path = tmp_path / "audit.json"
    audit_manifest_path.write_text(json.dumps(build_human_audit_manifest(cases)), encoding="utf-8")
    store = ReviewStore(packet_path, tmp_path / "progress.json", audit_manifest_path)
    assert store.status()["progress"] == "0 / 30"
    non_audit_id = next(
        case.case_id
        for case in cases
        if case.case_id not in set(json.loads(audit_manifest_path.read_text())["case_ids"])
    )
    store.save_decision(
        ReviewDecision(
            case_id=non_audit_id,
            outcome=ReviewOutcome.UNCERTAIN,
        )
    )
    assert store.status()["progress"] == "0 / 30"
    with pytest.raises(RuntimeError, match="30"):
        store.require_finalize_ready()


def test_all_30_frozen_audit_cases_are_required_to_finalize(tmp_path: Path) -> None:
    cases = _audit_cases()
    packet_path = tmp_path / "packet.json"
    prepare_review_packet(cases, packet_path, tmp_path / "manifest.json")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(build_human_audit_manifest(cases)), encoding="utf-8")
    store = ReviewStore(packet_path, tmp_path / "progress.json", audit_path)
    audit_ids = json.loads(audit_path.read_text())["case_ids"]
    for case_id in audit_ids[:-1]:
        store.save_decision(ReviewDecision(case_id=case_id, outcome=ReviewOutcome.UNCERTAIN))
    assert store.status()["progress"] == "29 / 30"
    with pytest.raises(RuntimeError, match="missing 1"):
        store.require_finalize_ready()
    store.save_decision(ReviewDecision(case_id=audit_ids[-1], outcome=ReviewOutcome.UNCERTAIN))
    assert store.status()["finalize_ready"] is True
    store.require_finalize_ready()


def test_ai_and_human_analysis_are_separately_labeled() -> None:
    ai_summary = aggregate_ai_classifications(_audit_cases())
    human_summary = aggregate_review_decisions(
        (
            ReviewDecision(
                case_id="audit-case-1",
                outcome=ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE,
            ),
        )
    )
    assert ai_summary["analysis_type"] == "AUTOMATED_DIAGNOSTIC_ANALYSIS"
    assert ai_summary["attempted_count"] == 120
    assert ai_summary["successful_count"] == 90
    assert ai_summary["unavailable_count"] == 30
    assert human_summary["analysis_type"] == "HUMAN_REVIEW_AUDIT"


def test_tracked_audit_manifest_references_the_unchanged_population() -> None:
    root = Path(__file__).parents[2]
    population = json.loads(
        (root / "data/manifests/evaluation/phase15_review_manifest.json").read_text()
    )
    audit = json.loads(
        (root / "data/manifests/evaluation/phase15_human_audit_manifest.json").read_text()
    )
    assert population["case_count"] == 120
    assert population["case_ids_sha256"] == (
        "8bc039f51344f3af47b817a5e1bbf51d4d087f49768ea5ca2b2ce3a29cd53777"
    )
    assert audit["count"] == 30
    assert audit["case_ids_sha256"] == (
        "4fc44ab5f5284ed720421dd13ef6f866e69292b5709b753c5464afacc6bd8af9"
    )
    assert audit["population_case_ids_sha256"] == population["case_ids_sha256"]
