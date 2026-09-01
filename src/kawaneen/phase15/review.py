"""Private 120-case packet and frozen 30-case human audit store."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from .contracts import PHASE15_SEED, ErrorCategory, ReviewCase, ReviewDecision, ReviewOutcome
from .evidence import write_json_atomic

REVIEW_CASE_COUNT = 120
HUMAN_AUDIT_CASE_COUNT = 30
HUMAN_AUDIT_SELECTION_SEED = PHASE15_SEED
HUMAN_AUDIT_STAGE_TARGETS = {
    "retrieval": 8,
    "normalization": 7,
    "generation": 7,
    "dialect": 4,
    "reranking": 2,
}
REVIEW_ROOT = Path("artifacts/private/phase15_evaluation/review")
PACKET_FILENAME = "review_packet.json"
PROGRESS_FILENAME = "review_progress.json"
AUDIT_MANIFEST_PATH = Path("data/manifests/evaluation/phase15_human_audit_manifest.json")


def _case_id_digest(case_ids: Iterable[str]) -> str:
    return sha256("\n".join(sorted(case_ids)).encode()).hexdigest()


def _stable_case_order(case_id: str, seed: int) -> str:
    return sha256(f"{seed}:{case_id}".encode()).hexdigest()


def _validate_review_population(cases: tuple[ReviewCase, ...]) -> None:
    if len(cases) != REVIEW_CASE_COUNT:
        raise ValueError("Phase 15 review population must contain exactly 120 cases")
    if len({case.case_id for case in cases}) != REVIEW_CASE_COUNT:
        raise ValueError("Phase 15 review population case IDs must be unique")
    if any(case.holdout for case in cases):
        raise ValueError("Phase 15 human audit cannot contain HOLDOUT cases")


def _diversity_pick(
    candidates: list[ReviewCase], count: int, selected: list[ReviewCase], seed: int
) -> list[ReviewCase]:
    """Pick deterministically while covering observable diagnostic strata."""

    remaining = list(candidates)
    chosen: list[ReviewCase] = []
    fields = ("language", "legal_category", "answerability", "severity")
    while remaining and len(chosen) < count:
        seen = {
            field: {str(getattr(case, field)) for case in (*selected, *chosen)} for field in fields
        }
        seen_ai = {case.ai_suggestion is not None for case in (*selected, *chosen)}

        def score(
            case: ReviewCase,
            *,
            seen: dict[str, set[str]] = seen,
            seen_ai: set[bool] = seen_ai,
        ) -> tuple[int, int, int, int, int, str]:
            ai_state = case.ai_suggestion is not None
            return (
                sum(str(getattr(case, field)) not in seen[field] for field in fields),
                int(ai_state not in seen_ai),
                int(case.answerability == "unanswerable"),
                int(case.severity == "high"),
                int(case.ai_suggestion is None),
                _stable_case_order(case.case_id, seed),
            )

        best = max(remaining, key=score)
        remaining.remove(best)
        chosen.append(best)
    return chosen


def select_human_audit_cases(
    cases: Iterable[ReviewCase], *, seed: int = HUMAN_AUDIT_SELECTION_SEED
) -> tuple[ReviewCase, ...]:
    """Select the pre-human, deterministic 30-case audit subset from the 120 DEV cases."""

    case_list = tuple(cases)
    _validate_review_population(case_list)
    selected: list[ReviewCase] = []
    for stage, target in HUMAN_AUDIT_STAGE_TARGETS.items():
        stage_cases = [case for case in case_list if case.pipeline_stage == stage]
        if len(stage_cases) < target:
            target = min(target, len(stage_cases))
        selected.extend(
            _diversity_pick(
                sorted(stage_cases, key=lambda case: _stable_case_order(case.case_id, seed)),
                target,
                selected,
                seed,
            )
        )

    remaining = [case for case in case_list if case not in selected]
    if len(selected) < HUMAN_AUDIT_CASE_COUNT:
        selected.extend(
            _diversity_pick(
                sorted(remaining, key=lambda case: _stable_case_order(case.case_id, seed + 1)),
                HUMAN_AUDIT_CASE_COUNT - len(selected),
                selected,
                seed + 1,
            )
        )
    if len(selected) != HUMAN_AUDIT_CASE_COUNT:
        raise ValueError("Phase 15 human audit selection must contain exactly 30 cases")
    return tuple(selected)


def build_human_audit_manifest(
    cases: Iterable[ReviewCase], *, seed: int = HUMAN_AUDIT_SELECTION_SEED
) -> dict[str, Any]:
    """Build text-free metadata for the frozen 30-case human audit subset."""

    case_list = tuple(cases)
    _validate_review_population(case_list)
    selected = select_human_audit_cases(case_list, seed=seed)

    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(getattr(case, field)) for case in selected).items()))

    return {
        "schema_version": "phase15-human-audit-v1",
        "count": HUMAN_AUDIT_CASE_COUNT,
        "case_ids": [case.case_id for case in selected],
        "case_ids_sha256": _case_id_digest(case.case_id for case in selected),
        "selection_seed": seed,
        "population_case_count": REVIEW_CASE_COUNT,
        "population_case_ids_sha256": _case_id_digest(case.case_id for case in case_list),
        "holdout_case_count": 0,
        "pipeline_stage_targets": dict(HUMAN_AUDIT_STAGE_TARGETS),
        "pipeline_stage_distribution": counts("pipeline_stage"),
        "language_distribution": counts("language"),
        "legal_category_distribution": counts("legal_category"),
        "answerability_distribution": counts("answerability"),
        "severity_distribution": counts("severity"),
        "ai_state_distribution": dict(
            sorted(
                Counter(
                    "successful" if case.ai_suggestion is not None else "unavailable"
                    for case in selected
                ).items()
            )
        ),
        "provenance": "PHASE15_DEV",
        "selection_type": "PRE_SELECTED_INDEPENDENT_HUMAN_AUDIT",
    }


def write_human_audit_manifest(
    cases: Iterable[ReviewCase], manifest_path: Path, *, seed: int = HUMAN_AUDIT_SELECTION_SEED
) -> dict[str, Any]:
    manifest = build_human_audit_manifest(cases, seed=seed)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("frozen human audit manifest cannot be changed")
        return existing
    write_json_atomic(manifest_path, manifest)
    return manifest


def build_review_manifest(cases: Iterable[ReviewCase]) -> dict[str, Any]:
    case_list = tuple(cases)
    if len(case_list) != REVIEW_CASE_COUNT:
        raise ValueError("Phase 15 review packet must contain exactly 120 cases")
    if len({case.case_id for case in case_list}) != REVIEW_CASE_COUNT:
        raise ValueError("Phase 15 review case IDs must be unique")
    if any(case.holdout for case in case_list):
        raise ValueError("Phase 15 review packet cannot contain HOLDOUT cases")

    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(getattr(case, field)) for case in case_list).items()))

    return {
        "schema_version": "phase15-review-v2",
        "case_count": REVIEW_CASE_COUNT,
        "case_ids_sha256": _case_id_digest(case.case_id for case in case_list),
        "holdout_case_count": 0,
        "language_distribution": counts("language"),
        "pipeline_stage_distribution": counts("pipeline_stage"),
        "legal_category_distribution": counts("legal_category"),
        "answerability_distribution": counts("answerability"),
        "severity_distribution": counts("severity"),
        "ai_preclassification_attempted": sum(
            case.ai_preclassification_attempted or case.ai_suggestion is not None
            for case in case_list
        ),
        "ai_preclassification_successful": sum(
            case.ai_suggestion is not None for case in case_list
        ),
        "ai_preclassification_unavailable": sum(
            case.ai_preclassification_attempted and case.ai_suggestion is None for case in case_list
        ),
        "provenance": "PHASE15_DEV",
    }


def prepare_review_packet(
    cases: Iterable[ReviewCase], packet_path: Path, manifest_path: Path
) -> dict[str, Any]:
    case_list = tuple(cases)
    manifest = build_review_manifest(case_list)
    packet = {
        "schema_version": "phase15-review-v2",
        "case_ids": [case.case_id for case in case_list],
        "cases": [case.model_dump(mode="json") for case in case_list],
    }
    write_json_atomic(packet_path, packet)
    write_json_atomic(manifest_path, manifest)
    return manifest


def aggregate_review_decisions(
    decisions: Iterable[ReviewDecision],
) -> dict[str, Any]:
    """Aggregate human outcomes, counting failure taxonomy only for confirmed failures."""

    decision_list = tuple(decisions)
    outcome_counts = Counter(decision.outcome.value for decision in decision_list)
    confirmed = tuple(
        decision
        for decision in decision_list
        if decision.outcome is ReviewOutcome.CONFIRMED_FAILURE
    )
    taxonomy = Counter(
        decision.primary.value for decision in confirmed if decision.primary is not None
    )
    return {
        "analysis_type": "HUMAN_REVIEW_AUDIT",
        "provenance": "HUMAN_REVIEWED_DIAGNOSTIC",
        "human_reviewed_count": len(decision_list),
        "review_outcome_distribution": dict(sorted(outcome_counts.items())),
        "confirmed_failure_count": len(confirmed),
        "confirmed_failure_taxonomy": dict(sorted(taxonomy.items())),
        "borderline_no_confirmed_failure_count": outcome_counts.get(
            ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE.value, 0
        ),
        "uncertain_count": outcome_counts.get(ReviewOutcome.UNCERTAIN.value, 0),
    }


def aggregate_ai_classifications(cases: Iterable[ReviewCase]) -> dict[str, Any]:
    """Summarize automated classifications separately from human audit decisions."""

    case_list = tuple(cases)
    successful = tuple(case for case in case_list if case.ai_suggestion is not None)
    attempted = tuple(
        case
        for case in case_list
        if case.ai_preclassification_attempted or case.ai_suggestion is not None
    )
    return {
        "analysis_type": "AUTOMATED_DIAGNOSTIC_ANALYSIS",
        "provenance": "PHASE15_DEV",
        "population_count": len(case_list),
        "attempted_count": len(attempted),
        "successful_count": len(successful),
        "unavailable_count": len(attempted) - len(successful),
        "primary_root_cause_distribution": dict(
            sorted(
                Counter(
                    case.ai_suggestion.value for case in case_list if case.ai_suggestion is not None
                ).items()
            )
        ),
    }


def ai_human_agreement(
    cases: Iterable[ReviewCase], decisions: Iterable[ReviewDecision]
) -> dict[str, Any]:
    """Compare AI suggestions only with confirmed human root-cause labels."""

    ai_by_case = {
        case.case_id: case.ai_suggestion for case in cases if case.ai_suggestion is not None
    }
    eligible = tuple(
        decision
        for decision in decisions
        if decision.outcome is ReviewOutcome.CONFIRMED_FAILURE
        and decision.primary is not None
        and decision.case_id in ai_by_case
    )
    agreement_count = sum(ai_by_case[decision.case_id] is decision.primary for decision in eligible)
    disagreement_counts: Counter[str] = Counter()
    for decision in eligible:
        ai_category = ai_by_case[decision.case_id]
        if decision.primary is not None and ai_category is not decision.primary:
            disagreement_counts[f"{ai_category.value} -> {decision.primary.value}"] += 1
    per_category: dict[str, dict[str, int]] = {}
    for category in ErrorCategory:
        ai_count = sum(ai_by_case[decision.case_id] is category for decision in eligible)
        human_count = sum(decision.primary is category for decision in eligible)
        agreement = sum(
            ai_by_case[decision.case_id] is category and decision.primary is category
            for decision in eligible
        )
        if ai_count or human_count:
            per_category[category.value] = {
                "ai_count": ai_count,
                "human_count": human_count,
                "exact_agreement_count": agreement,
            }
    if eligible:
        ai_marginals = Counter(ai_by_case[decision.case_id] for decision in eligible)
        human_marginals = Counter(decision.primary for decision in eligible)
        expected_agreement = (
            sum(ai_marginals[category] * human_marginals[category] for category in ErrorCategory)
            / len(eligible) ** 2
        )
        observed_agreement = agreement_count / len(eligible)
        denominator = 1.0 - expected_agreement
        cohens_kappa = (
            (observed_agreement - expected_agreement) / denominator if denominator else None
        )
    else:
        cohens_kappa = None
    return {
        "eligible_count": len(eligible),
        "agreement_count": agreement_count,
        "agreement_rate": agreement_count / len(eligible) if eligible else None,
        "cohens_kappa": cohens_kappa,
        "per_category": per_category,
        "disagreement_counts": dict(sorted(disagreement_counts.items())),
    }


class ReviewStore:
    """Atomic progress store keyed by immutable packet case IDs."""

    def __init__(self, packet_path: Path, progress_path: Path, audit_manifest_path: Path) -> None:
        self.packet_path = packet_path
        self.progress_path = progress_path
        self.audit_manifest_path = audit_manifest_path

    def cases(self) -> tuple[ReviewCase, ...]:
        payload = json.loads(self.packet_path.read_text(encoding="utf-8"))
        cases = tuple(ReviewCase.model_validate(item) for item in payload.get("cases", ()))
        if (
            len(cases) != REVIEW_CASE_COUNT
            or len({case.case_id for case in cases}) != REVIEW_CASE_COUNT
        ):
            raise ValueError("review packet must contain exactly 120 unique cases")
        return cases

    def _decisions(self) -> dict[str, ReviewDecision]:
        if not self.progress_path.exists():
            return {}
        payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
        return {
            case_id: ReviewDecision.model_validate(decision)
            for case_id, decision in payload.get("decisions", {}).items()
        }

    def audit_case_ids(self) -> frozenset[str]:
        payload = json.loads(self.audit_manifest_path.read_text(encoding="utf-8"))
        case_ids = tuple(str(case_id) for case_id in payload.get("case_ids", ()))
        if payload.get("count") != HUMAN_AUDIT_CASE_COUNT:
            raise ValueError("human audit manifest must contain exactly 30 cases")
        if len(case_ids) != HUMAN_AUDIT_CASE_COUNT or len(set(case_ids)) != HUMAN_AUDIT_CASE_COUNT:
            raise ValueError("human audit manifest case IDs must be unique and exactly 30")
        cases = self.cases()
        population_hash = _case_id_digest(case.case_id for case in cases)
        if payload.get("population_case_ids_sha256") != population_hash:
            raise ValueError("human audit manifest does not match the frozen 120-case population")
        valid_ids = {case.case_id for case in cases}
        if not set(case_ids) <= valid_ids:
            raise ValueError("human audit manifest contains an unknown or non-DEV case ID")
        if payload.get("case_ids_sha256") != _case_id_digest(case_ids):
            raise ValueError("human audit manifest case-ID hash is invalid")
        selection_seed = payload.get("selection_seed")
        if not isinstance(selection_seed, int):
            raise ValueError("human audit manifest selection seed is required")
        expected_ids = {
            case.case_id for case in select_human_audit_cases(cases, seed=selection_seed)
        }
        if set(case_ids) != expected_ids:
            raise ValueError("human audit manifest does not match deterministic selection")
        return frozenset(case_ids)

    def audit_cases(self) -> tuple[ReviewCase, ...]:
        audit_ids = self.audit_case_ids()
        return tuple(case for case in self.cases() if case.case_id in audit_ids)

    def save_decision(self, decision: ReviewDecision) -> None:
        valid_ids = {case.case_id for case in self.cases()}
        if decision.case_id not in valid_ids:
            raise ValueError(f"unknown immutable review case ID: {decision.case_id}")
        decisions = self._decisions()
        decisions[decision.case_id] = decision
        write_json_atomic(
            self.progress_path,
            {
                "schema_version": "phase15-review-progress-v2",
                "decisions": {
                    key: value.model_dump(mode="json") for key, value in sorted(decisions.items())
                },
            },
        )

    def reviewed_count(self) -> int:
        audit_ids = self.audit_case_ids()
        return len(audit_ids & self._decisions().keys())

    def decision_for(self, case_id: str) -> ReviewDecision | None:
        return self._decisions().get(case_id)

    def status(self) -> dict[str, Any]:
        reviewed = self.reviewed_count()
        total_packet_reviewed = len(self._decisions())
        return {
            "reviewed": reviewed,
            "total": HUMAN_AUDIT_CASE_COUNT,
            "remaining": HUMAN_AUDIT_CASE_COUNT - reviewed,
            "progress": f"{reviewed} / {HUMAN_AUDIT_CASE_COUNT}",
            "non_audit_reviewed": total_packet_reviewed - reviewed,
            "packet_path": self.packet_path.as_posix(),
            "progress_path": self.progress_path.as_posix(),
            "finalize_ready": reviewed == HUMAN_AUDIT_CASE_COUNT,
        }

    def next_unreviewed(self) -> ReviewCase | None:
        decisions = self._decisions()
        return next((case for case in self.audit_cases() if case.case_id not in decisions), None)

    def require_finalize_ready(self) -> None:
        audit_ids = self.audit_case_ids()
        missing = audit_ids - self._decisions().keys()
        if missing:
            raise RuntimeError(
                "phase15 finalize requires all 30 frozen human-audit cases; "
                f"missing {len(missing)} decisions"
            )


def default_review_paths(root: Path = Path(".")) -> tuple[Path, Path, Path]:
    private_root = root / REVIEW_ROOT
    return (
        private_root / PACKET_FILENAME,
        private_root / PROGRESS_FILENAME,
        root / "data/manifests/evaluation/phase15_review_manifest.json",
    )


def default_audit_manifest_path(root: Path = Path(".")) -> Path:
    return root / AUDIT_MANIFEST_PATH
