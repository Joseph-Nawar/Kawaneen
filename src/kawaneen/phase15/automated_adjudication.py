"""Deterministic two-pass adjudication for the frozen Phase 15 audit subset.

This is a diagnostic workflow, not human review and not a production decision
path.  The outcome rules deliberately use the case evidence and diagnostics,
never the prior model category, so the initial model is only a comparison
source in the resulting aggregate.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .contracts import ErrorCategory, ReviewCase, ReviewOutcome
from .evidence import write_json_atomic

AUDIT_CASE_COUNT = 30
AUTOMATED_ADJUDICATION_SCHEMA = "phase15-automated-adjudication-v1"
AUTOMATED_ADJUDICATION_PATH = Path(
    "artifacts/private/phase15_evaluation/review/phase15_30_case_automated_adjudication.json"
)
AUTOMATED_AUDIT_SUMMARY_PATH = Path("data/evaluation/phase15_automated_audit_summary.json")
VALID_OUTCOMES = frozenset(outcome.value for outcome in ReviewOutcome)
VALID_CATEGORIES = frozenset(category.value for category in ErrorCategory)


def _text(value: object) -> str:
    return str(value).strip().lower() if value is not None else ""


def _evidence_fragments(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split("\n") if part.strip()][:2]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = cast(Sequence[object], value)
        return [str(part).strip() for part in parts if str(part).strip()][:2]
    return [str(value)]


def _classification(case: ReviewCase) -> tuple[ReviewOutcome, ErrorCategory | None, int, str]:
    """Apply the frozen evidence-first diagnostic rules to one DEV case."""

    diagnostics = case.diagnostics
    system = _text(diagnostics.get("system"))
    stage = _text(case.pipeline_stage)
    answerability = _text(case.answerability)

    if stage == "reranking" and diagnostics.get("before_recall", 0) > diagnostics.get(
        "after_recall", 0
    ):
        return (
            ReviewOutcome.CONFIRMED_FAILURE,
            ErrorCategory.RERANKER_FAILURE,
            5,
            (
                "The persisted DEV diagnostic shows relevant evidence before reranking "
                "and absent after reranking."
            ),
        )

    if stage == "generation":
        if answerability == "unanswerable" and diagnostics.get("decision") == "answer":
            return (
                ReviewOutcome.CONFIRMED_FAILURE,
                ErrorCategory.GENERATOR_HALLUCINATION,
                5,
                "The persisted generator decision answered an explicitly unanswerable DEV query.",
            )
        return (
            ReviewOutcome.UNCERTAIN,
            None,
            2,
            (
                "The persisted generator output is malformed or abstains; the taxonomy "
                "does not support assigning a more specific root cause from this evidence alone."
            ),
        )

    if stage == "normalization":
        return (
            ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE,
            None,
            3,
            (
                "The diagnostic records a normalization-induced ranking change, but no "
                "persisted evidence establishes a loss of answer-supporting retrieval."
            ),
        )

    if stage == "dialect":
        systems = {_text(item) for item in diagnostics.get("systems", ())}
        if systems == {"bm25"}:
            return (
                ReviewOutcome.CONFIRMED_FAILURE,
                ErrorCategory.LEXICAL_MISMATCH,
                4,
                (
                    "The dialect diagnostic reports BM25 degradation; the earliest supported "
                    "mechanism is a lexical mismatch between the dialect query and corpus wording."
                ),
            )
        return (
            ReviewOutcome.CONFIRMED_FAILURE,
            ErrorCategory.SEMANTIC_RETRIEVAL_FAILURE,
            4,
            (
                "The dialect diagnostic reports dense or hybrid retrieval degradation while "
                "the matched evidence target remains available."
            ),
        )

    if stage == "retrieval":
        if not case.evidence_text:
            return (
                ReviewOutcome.UNCERTAIN,
                None,
                2,
                (
                    "No expected evidence is persisted for this DEV case, so a retrieval root "
                    "cause cannot be distinguished from a missing-source condition."
                ),
            )
        if diagnostics.get("relevant_rank") is not None:
            return (
                ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE,
                None,
                3,
                (
                    "Relevant evidence is persisted within the reported top-10 boundary; the "
                    "diagnostic is a borderline ranking case rather than a confirmed "
                    "retrieval failure."
                ),
            )
        if system == "bm25":
            return (
                ReviewOutcome.CONFIRMED_FAILURE,
                ErrorCategory.LEXICAL_MISMATCH,
                4,
                (
                    "The persisted BM25 miss has a retained expected-evidence target; the "
                    "earliest supported mechanism is lexical mismatch."
                ),
            )
        return (
            ReviewOutcome.CONFIRMED_FAILURE,
            ErrorCategory.SEMANTIC_RETRIEVAL_FAILURE,
            4,
            (
                "The persisted dense or hybrid miss has a retained expected-evidence target "
                "absent from the reported top-10 results."
            ),
        )

    return (
        ReviewOutcome.UNCERTAIN,
        None,
        2,
        "The available DEV diagnostics do not establish a unique earliest root cause.",
    )


def _critic(case: ReviewCase) -> tuple[ReviewOutcome, ErrorCategory | None, int, str | None]:
    """Independently inspect evidence for an earlier or unsupported cause."""

    diagnostics = case.diagnostics
    stage = _text(case.pipeline_stage)
    answerability = _text(case.answerability)
    system = _text(diagnostics.get("system"))
    if stage == "reranking" and diagnostics.get("before_recall", 0) > diagnostics.get(
        "after_recall", 0
    ):
        return ReviewOutcome.CONFIRMED_FAILURE, ErrorCategory.RERANKER_FAILURE, 5, None
    if (
        stage == "generation"
        and answerability == "unanswerable"
        and diagnostics.get("decision") == "answer"
    ):
        return ReviewOutcome.CONFIRMED_FAILURE, ErrorCategory.GENERATOR_HALLUCINATION, 5, None
    if stage == "generation":
        return ReviewOutcome.UNCERTAIN, None, 2, None
    if stage == "normalization":
        return ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE, None, 3, None
    if stage == "dialect":
        systems = {_text(item) for item in diagnostics.get("systems", ())}
        if systems == {"bm25"}:
            return ReviewOutcome.CONFIRMED_FAILURE, ErrorCategory.LEXICAL_MISMATCH, 4, None
        return ReviewOutcome.CONFIRMED_FAILURE, ErrorCategory.SEMANTIC_RETRIEVAL_FAILURE, 4, None
    if stage == "retrieval":
        if not case.evidence_text:
            return ReviewOutcome.UNCERTAIN, None, 2, None
        if diagnostics.get("relevant_rank") is not None:
            return ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE, None, 3, None
        if system == "bm25":
            return ReviewOutcome.CONFIRMED_FAILURE, ErrorCategory.LEXICAL_MISMATCH, 4, None
        return ReviewOutcome.CONFIRMED_FAILURE, ErrorCategory.SEMANTIC_RETRIEVAL_FAILURE, 4, None
    return ReviewOutcome.UNCERTAIN, None, 2, "no independently supported root cause"


def adjudicate_case(case: ReviewCase) -> dict[str, Any]:
    """Create one private, evidence-bearing two-pass adjudication record."""

    proposal, proposal_category, proposal_confidence, rationale = _classification(case)
    critic_outcome, critic_category, critic_confidence, concern = _critic(case)
    agreed = proposal is critic_outcome and proposal_category is critic_category

    if agreed:
        outcome, category, confidence = proposal, proposal_category, proposal_confidence
    else:
        outcome, category, confidence = (
            ReviewOutcome.UNCERTAIN,
            None,
            min(proposal_confidence, critic_confidence),
        )
        rationale = (
            "The adjudicator and critic passes disagree, so the reconciled outcome is UNCERTAIN."
        )
        concern = concern or "adjudicator and critic passes disagreed"

    evidence = _evidence_fragments(case.evidence_text)
    key_evidence = [f"expected evidence: {fragment}" for fragment in evidence]
    if not key_evidence:
        key_evidence.append("expected evidence: none persisted for this DEV case")
    if case.diagnostics.get("trigger"):
        key_evidence.append(f"diagnostic trigger: {case.diagnostics['trigger']}")

    prior_category = case.ai_suggestion.value if case.ai_suggestion is not None else None
    return {
        "case_id": case.case_id,
        "pipeline_stage": case.pipeline_stage,
        "language": case.language,
        "legal_category": case.legal_category,
        "answerability": case.answerability,
        "severity": case.severity,
        "holdout": case.holdout,
        "query": case.query_text,
        "expected_evidence": case.evidence_text,
        "diagnostics": dict(case.diagnostics),
        "prior_ai_preclassification": {
            "status": "available" if prior_category is not None else "unavailable",
            "category": prior_category,
        },
        "passes": {
            "adjudicator": {
                "outcome": proposal.value,
                "primary_category": proposal_category.value if proposal_category else None,
                "confidence": proposal_confidence,
                "rationale": rationale,
            },
            "critic": {
                "outcome": critic_outcome.value,
                "primary_category": critic_category.value if critic_category else None,
                "confidence": critic_confidence,
                "concern": concern,
            },
        },
        "adjudication": {
            "outcome": outcome.value,
            "primary_category": category.value if category else None,
            "secondary_category": None,
            "confidence": confidence,
            "rationale": rationale,
            "key_evidence": key_evidence,
        },
        "critic": {"agreed": agreed, "concern": concern},
    }


def adjudicate_cases(cases: Iterable[ReviewCase]) -> list[dict[str, Any]]:
    case_list = tuple(cases)
    if (
        len(case_list) != AUDIT_CASE_COUNT
        or len({case.case_id for case in case_list}) != AUDIT_CASE_COUNT
    ):
        raise ValueError("automated adjudication requires exactly 30 unique DEV cases")
    if any(case.holdout for case in case_list):
        raise ValueError("automated adjudication cannot include HOLDOUT cases")
    return [adjudicate_case(case) for case in case_list]


def build_automated_adjudication(
    cases: Iterable[ReviewCase], *, population_hash: str, audit_hash: str
) -> dict[str, Any]:
    case_records = adjudicate_cases(cases)
    summary = aggregate_automated_audit(
        case_records, audit_hash=audit_hash, population_hash=population_hash
    )
    return {
        "schema_version": AUTOMATED_ADJUDICATION_SCHEMA,
        "population_hash": population_hash,
        "audit_hash": audit_hash,
        "methodology": {
            "label": "AUTOMATED_ADJUDICATION_DIAGNOSTIC",
            "description": (
                "Frozen 30-case automated adjudication audit; no expert or human gold "
                "adjudication was performed."
            ),
            "workflow": (
                "fixed local deterministic evidence workflow with independent adjudicator "
                "and critic passes"
            ),
            "primary_rule": (
                "earliest demonstrable root cause; UNCERTAIN when evidence is insufficient"
            ),
            "human_labels_present": False,
        },
        "summary": summary,
        "cases": case_records,
    }


def aggregate_automated_audit(
    case_records: Iterable[Mapping[str, Any]], *, audit_hash: str, population_hash: str
) -> dict[str, Any]:
    records = tuple(case_records)
    outcomes = Counter(str(record["adjudication"]["outcome"]) for record in records)
    confirmed = tuple(
        record
        for record in records
        if record["adjudication"]["outcome"] == ReviewOutcome.CONFIRMED_FAILURE.value
    )
    taxonomy = Counter(str(record["adjudication"]["primary_category"]) for record in confirmed)
    confidence = Counter(str(record["adjudication"]["confidence"]) for record in records)
    comparable = tuple(
        record
        for record in records
        if record["prior_ai_preclassification"]["status"] == "available"
    )
    agreement = sum(
        record["prior_ai_preclassification"]["category"]
        == record["adjudication"]["primary_category"]
        for record in comparable
    )
    ai_unavailable = sum(
        record["prior_ai_preclassification"]["status"] == "unavailable" for record in records
    )
    return {
        "schema_version": "phase15-automated-audit-summary-v1",
        "methodology_label": "AUTOMATED_ADJUDICATION_DIAGNOSTIC",
        "provenance": "PHASE15_DEV",
        "audit_count": len(records),
        "audit_population_hash": audit_hash,
        "population_hash": population_hash,
        "outcome_distribution": dict(sorted(outcomes.items())),
        "confirmed_failure_count": len(confirmed),
        "confirmed_failure_taxonomy": dict(sorted(taxonomy.items())),
        "borderline_count": outcomes.get(ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE.value, 0),
        "uncertain_count": outcomes.get(ReviewOutcome.UNCERTAIN.value, 0),
        "confidence_distribution": dict(sorted(confidence.items())),
        "initial_model_vs_adjudication_workflow_agreement": {
            "comparable_count": len(comparable),
            "agreement_count": agreement,
            "agreement_rate": agreement / len(comparable) if comparable else None,
        },
        "prior_ai_unavailable_count": ai_unavailable,
        "human_labels_present": False,
    }


def validate_automated_adjudication(
    private_path: Path,
    aggregate_path: Path,
    *,
    expected_case_ids: set[str],
    expected_population_hash: str,
    expected_audit_hash: str,
) -> dict[str, Any]:
    payload = json.loads(private_path.read_text(encoding="utf-8"))
    records = payload.get("cases", ())
    if payload.get("schema_version") != AUTOMATED_ADJUDICATION_SCHEMA:
        raise ValueError("automated adjudication schema is invalid")
    if payload.get("population_hash") != expected_population_hash:
        raise ValueError("automated adjudication population hash is invalid")
    if payload.get("audit_hash") != expected_audit_hash:
        raise ValueError("automated adjudication audit hash is invalid")
    if len(records) != AUDIT_CASE_COUNT:
        raise ValueError("automated adjudication must contain exactly 30 cases")
    actual_ids = {str(record.get("case_id")) for record in records}
    if actual_ids != expected_case_ids:
        raise ValueError("automated adjudication case IDs do not match the frozen audit subset")
    for record in records:
        if record.get("holdout") is not False:
            raise ValueError("automated adjudication contains a HOLDOUT case")
        adjudication = record.get("adjudication", {})
        outcome = adjudication.get("outcome")
        category = adjudication.get("primary_category")
        if outcome not in VALID_OUTCOMES:
            raise ValueError("automated adjudication contains an invalid outcome")
        if category is not None and category not in VALID_CATEGORIES:
            raise ValueError("automated adjudication contains an invalid root-cause category")
        if outcome == ReviewOutcome.CONFIRMED_FAILURE.value and category is None:
            raise ValueError("confirmed automated failure requires a primary category")
        if outcome != ReviewOutcome.CONFIRMED_FAILURE.value and category is not None:
            raise ValueError("non-confirmed automated outcome cannot have a primary category")
        if not isinstance(adjudication.get("confidence"), int) or not (
            1 <= adjudication["confidence"] <= 5
        ):
            raise ValueError("automated adjudication confidence must be an integer from 1 to 5")

    summary = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if summary.get("methodology_label") != "AUTOMATED_ADJUDICATION_DIAGNOSTIC":
        raise ValueError("automated audit summary has the wrong methodology label")
    if summary.get("audit_count") != AUDIT_CASE_COUNT:
        raise ValueError("automated audit summary must contain exactly 30 cases")
    if summary.get("audit_population_hash") != expected_audit_hash:
        raise ValueError("automated audit summary audit hash is invalid")
    if summary.get("population_hash") != expected_population_hash:
        raise ValueError("automated audit summary population hash is invalid")
    if any(key in summary for key in ("query", "query_text", "evidence", "expected_evidence")):
        raise ValueError("tracked automated audit summary must be text-free")
    return {"case_count": len(records), "audit_hash": expected_audit_hash}


def write_automated_audit_artifacts(
    cases: Iterable[ReviewCase],
    *,
    root: Path,
    population_hash: str,
    audit_hash: str,
) -> tuple[Path, Path]:
    payload = build_automated_adjudication(
        cases, population_hash=population_hash, audit_hash=audit_hash
    )
    private_path = root / AUTOMATED_ADJUDICATION_PATH
    aggregate_path = root / AUTOMATED_AUDIT_SUMMARY_PATH
    write_json_atomic(private_path, payload)
    aggregate = aggregate_automated_audit(
        payload["cases"], audit_hash=audit_hash, population_hash=population_hash
    )
    write_json_atomic(aggregate_path, aggregate)
    return private_path, aggregate_path
