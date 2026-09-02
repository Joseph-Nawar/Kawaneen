from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawaneen.phase15.automated_adjudication import (
    adjudicate_case,
    aggregate_automated_audit,
    build_automated_adjudication,
    validate_automated_adjudication,
    write_automated_audit_artifacts,
)
from kawaneen.phase15.contracts import ErrorCategory, ReviewCase, ReviewOutcome


def _case(**updates: object) -> ReviewCase:
    values: dict[str, object] = {
        "case_id": "dev-case",
        "language": "ar",
        "pipeline_stage": "retrieval",
        "legal_category": "deadline",
        "answerability": "answerable",
        "severity": "medium",
        "query_text": "ما الموعد؟",
        "evidence_text": "الموعد هو عشرة أيام.",
        "diagnostics": {
            "system": "bm25",
            "relevant_rank": None,
            "trigger": "bm25 miss: relevant evidence absent from top10",
        },
    }
    values.update(updates)
    return ReviewCase.model_validate(values)


def test_adjudication_uses_two_passes_and_does_not_copy_ai_suggestion() -> None:
    case = _case(ai_suggestion=ErrorCategory.NORMALIZATION_FAILURE)
    result = adjudicate_case(case)
    assert result["adjudication"]["outcome"] == ReviewOutcome.CONFIRMED_FAILURE.value
    assert result["adjudication"]["primary_category"] == ErrorCategory.LEXICAL_MISMATCH.value
    assert result["prior_ai_preclassification"]["category"] == (
        ErrorCategory.NORMALIZATION_FAILURE.value
    )
    assert result["passes"]["adjudicator"]["primary_category"] == (
        ErrorCategory.LEXICAL_MISMATCH.value
    )
    assert "agreed" in result["consistency_check"]


def test_adjudication_can_be_borderline_or_uncertain_without_forced_category() -> None:
    borderline = adjudicate_case(
        _case(
            pipeline_stage="normalization",
            diagnostics={"normalization": "light", "trigger": "changed top10 behavior"},
        )
    )
    assert borderline["adjudication"]["outcome"] == (
        ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE.value
    )
    assert borderline["adjudication"]["primary_category"] is None
    assert borderline["adjudication"]["failure_mode"] is None

    contract_failure = adjudicate_case(
        _case(
            pipeline_stage="generation",
            answerability="unanswerable",
            evidence_text=None,
            diagnostics={"parsed_decision": "invalid", "trigger": "malformed output"},
        )
    )
    assert contract_failure["adjudication"]["outcome"] == ReviewOutcome.CONFIRMED_FAILURE.value
    assert contract_failure["adjudication"]["primary_category"] is None
    assert contract_failure["adjudication"]["failure_mode"] == "INVALID_GENERATION_CONTRACT"


def test_generation_contract_failure_is_not_added_to_root_cause_taxonomy() -> None:
    records = [
        adjudicate_case(
            _case(
                case_id="contract",
                pipeline_stage="generation",
                diagnostics={"parsed_decision": "invalid"},
            )
        ),
        adjudicate_case(_case(case_id="lexical")),
    ]
    summary = aggregate_automated_audit(records, audit_hash="audit", population_hash="population")
    assert summary["outcome_distribution"][ReviewOutcome.CONFIRMED_FAILURE.value] == 2
    assert summary["confirmed_failure_taxonomy"] == {
        ErrorCategory.LEXICAL_MISMATCH.value: 1,
    }
    assert summary["non_taxonomy_failure_modes"] == {"INVALID_GENERATION_CONTRACT": 1}


def test_aggregate_is_text_free_and_labels_workflow_agreement() -> None:
    cases = [
        adjudicate_case(_case(case_id="one", ai_suggestion=ErrorCategory.LEXICAL_MISMATCH)),
        adjudicate_case(
            _case(
                case_id="two",
                pipeline_stage="generation",
                diagnostics={"claims": True, "decision": "answer", "trigger": "unanswerable"},
                answerability="unanswerable",
                query_text="private query",
                evidence_text=None,
            )
        ),
    ]
    summary = aggregate_automated_audit(cases, audit_hash="audit", population_hash="population")
    assert summary["methodology_label"] == "AUTOMATED_ADJUDICATION_DIAGNOSTIC"
    assert (
        summary["initial_model_vs_rule_based_audit_category_agreement"]["comparable_category_count"]
        == 1
    )
    encoded = json.dumps(summary, ensure_ascii=False)
    assert "private query" not in encoded
    assert "private evidence" not in encoded


def test_agreement_excludes_null_and_non_taxonomy_final_categories() -> None:
    records = [
        adjudicate_case(_case(case_id="match", ai_suggestion=ErrorCategory.LEXICAL_MISMATCH)),
        adjudicate_case(
            _case(
                case_id="contract",
                pipeline_stage="generation",
                ai_suggestion=ErrorCategory.LEXICAL_MISMATCH,
                diagnostics={"parsed_decision": "invalid"},
            )
        ),
        adjudicate_case(_case(case_id="unavailable", ai_suggestion=None)),
    ]
    agreement = aggregate_automated_audit(
        records, audit_hash="audit", population_hash="population"
    )["initial_model_vs_rule_based_audit_category_agreement"]
    assert agreement["comparable_category_count"] == 1
    assert agreement["agreement_count"] == 1
    assert agreement["agreement_rate"] == 1.0
    assert agreement["disagreement_counts"] == {}


def test_validate_automated_adjudication_requires_exact_frozen_audit(tmp_path: Path) -> None:
    cases = [adjudicate_case(_case(case_id=f"case-{i}")) for i in range(30)]
    payload = {
        "schema_version": "phase15-automated-adjudication-v1",
        "population_hash": "population",
        "audit_hash": "audit",
        "cases": cases,
    }
    private_path = tmp_path / "adjudication.json"
    private_path.write_text(json.dumps(payload), encoding="utf-8")
    aggregate_path = tmp_path / "summary.json"
    aggregate_path.write_text(
        json.dumps(
            aggregate_automated_audit(cases, audit_hash="audit", population_hash="population")
        ),
        encoding="utf-8",
    )
    assert (
        validate_automated_adjudication(
            private_path,
            aggregate_path,
            expected_case_ids={f"case-{i}" for i in range(30)},
            expected_population_hash="population",
            expected_audit_hash="audit",
        )["case_count"]
        == 30
    )

    payload["cases"] = cases[:-1]
    private_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 30"):
        validate_automated_adjudication(
            private_path,
            aggregate_path,
            expected_case_ids={f"case-{i}" for i in range(30)},
            expected_population_hash="population",
            expected_audit_hash="audit",
        )


def test_writer_creates_private_two_pass_file_and_tracked_aggregate(tmp_path: Path) -> None:
    cases = tuple(_case(case_id=f"case-{i}") for i in range(30))
    payload = build_automated_adjudication(cases, population_hash="population", audit_hash="audit")
    assert payload["summary"]["audit_count"] == 30
    private_path, aggregate_path = write_automated_audit_artifacts(
        cases,
        root=tmp_path,
        population_hash="population",
        audit_hash="audit",
    )
    assert private_path.is_file()
    assert aggregate_path.is_file()
    validated = validate_automated_adjudication(
        private_path,
        aggregate_path,
        expected_case_ids={f"case-{i}" for i in range(30)},
        expected_population_hash="population",
        expected_audit_hash="audit",
    )
    assert validated["case_count"] == 30


def test_validator_rejects_id_mismatch_and_holdout_records(tmp_path: Path) -> None:
    cases = [adjudicate_case(_case(case_id=f"case-{i}")) for i in range(30)]
    private_path = tmp_path / "adjudication.json"
    aggregate_path = tmp_path / "summary.json"
    private_path.write_text(
        json.dumps(
            {
                "schema_version": "phase15-automated-adjudication-v1",
                "population_hash": "population",
                "audit_hash": "audit",
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    aggregate_path.write_text(
        json.dumps(
            aggregate_automated_audit(cases, audit_hash="audit", population_hash="population")
        ),
        encoding="utf-8",
    )
    cases[0]["case_id"] = "not-in-audit"
    private_path.write_text(
        json.dumps(
            {
                "schema_version": "phase15-automated-adjudication-v1",
                "population_hash": "population",
                "audit_hash": "audit",
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="case IDs"):
        validate_automated_adjudication(
            private_path,
            aggregate_path,
            expected_case_ids={f"case-{i}" for i in range(30)},
            expected_population_hash="population",
            expected_audit_hash="audit",
        )
    cases[0]["case_id"] = "case-0"
    cases[0]["holdout"] = True
    private_path.write_text(
        json.dumps(
            {
                "schema_version": "phase15-automated-adjudication-v1",
                "population_hash": "population",
                "audit_hash": "audit",
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="HOLDOUT"):
        validate_automated_adjudication(
            private_path,
            aggregate_path,
            expected_case_ids={f"case-{i}" for i in range(30)},
            expected_population_hash="population",
            expected_audit_hash="audit",
        )
