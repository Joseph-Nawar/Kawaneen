from __future__ import annotations

import pytest
from pydantic import ValidationError

from kawaneen.phase15.contracts import (
    ARABIC_EMBEDDING_MODEL,
    ErrorCategory,
    ExperimentPlan,
    GeneratorSubsetManifest,
    ProvenanceLabel,
    ReviewDecision,
    ReviewOutcome,
)


def test_experiment_plan_has_frozen_governance_defaults() -> None:
    plan = ExperimentPlan(
        base_sha="03f58284426c84c6c813be2b1e1bbbbbfd1c9a2d",
        research_questions=("rq1", "rq2", "rq3", "rq4", "rq5", "rq6", "rq7"),
        hard_prohibitions=("no holdout",),
    )

    assert plan.seed == 20260826
    assert plan.bootstrap_replicates == 2000
    assert plan.confidence == 0.95
    assert plan.arabic_embedding_model == ARABIC_EMBEDDING_MODEL
    assert plan.provenance is ProvenanceLabel.PHASE15_DEV


def test_unknown_provenance_and_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentPlan(
            base_sha="03f58284426c84c6c813be2b1e1bbbbbfd1c9a2d",
            research_questions=("rq1",) * 7,
            hard_prohibitions=("no holdout",),
            provenance="made_up",
        )

    with pytest.raises(ValidationError):
        ExperimentPlan(
            base_sha="03f58284426c84c6c813be2b1e1bbbbbfd1c9a2d",
            research_questions=("rq1",) * 7,
            hard_prohibitions=("no holdout",),
            unexpected=True,
        )


def test_invalid_bootstrap_settings_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentPlan(
            base_sha="03f58284426c84c6c813be2b1e1bbbbbfd1c9a2d",
            research_questions=("rq1",) * 7,
            hard_prohibitions=("no holdout",),
            bootstrap_replicates=1999,
        )


def test_generator_subset_requires_exact_phase15_counts() -> None:
    with pytest.raises(ValidationError):
        GeneratorSubsetManifest(
            answerable_gold_present_ids=tuple(f"p{i}" for i in range(30)),
            answerable_gold_absent_ids=tuple(f"a{i}" for i in range(30)),
            unanswerable_ids=tuple(f"u{i}" for i in range(19)),
        )


def test_review_decision_requires_review_outcome() -> None:
    with pytest.raises(ValidationError):
        ReviewDecision(case_id="case-1", primary=None)

    decision = ReviewDecision(
        case_id="case-1",
        primary=ErrorCategory.OCR_FAILURE,
        outcome=ReviewOutcome.CONFIRMED_FAILURE,
        confidence=2,
    )
    assert decision.primary is ErrorCategory.OCR_FAILURE


def test_confirmed_failure_requires_primary_category() -> None:
    with pytest.raises(ValidationError):
        ReviewDecision(
            case_id="case-1",
            outcome=ReviewOutcome.CONFIRMED_FAILURE,
            primary=None,
        )


def test_borderline_and_uncertain_allow_null_primary() -> None:
    borderline = ReviewDecision(
        case_id="case-1",
        outcome=ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE,
        primary=None,
    )
    uncertain = ReviewDecision(
        case_id="case-2",
        outcome=ReviewOutcome.UNCERTAIN,
        primary=None,
    )
    assert borderline.primary is None
    assert uncertain.primary is None


def test_borderline_rejects_failure_category() -> None:
    with pytest.raises(ValidationError):
        ReviewDecision(
            case_id="case-1",
            outcome=ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE,
            primary=ErrorCategory.OCR_FAILURE,
        )


def test_secondary_category_requires_primary_category() -> None:
    with pytest.raises(ValidationError):
        ReviewDecision(
            case_id="case-1",
            outcome=ReviewOutcome.UNCERTAIN,
            secondary=ErrorCategory.OCR_FAILURE,
        )
