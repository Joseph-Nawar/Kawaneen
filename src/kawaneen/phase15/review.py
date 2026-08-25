"""Private 120-case review packet and resumable atomic adjudication store."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import ReviewCase, ReviewDecision
from .evidence import write_json_atomic

REVIEW_CASE_COUNT = 120
MINIMUM_FINAL_REVIEWS = 100
REVIEW_ROOT = Path("artifacts/private/phase15_evaluation/review")
PACKET_FILENAME = "review_packet.json"
PROGRESS_FILENAME = "review_progress.json"


def _case_id_digest(case_ids: Iterable[str]) -> str:
    return sha256("\n".join(sorted(case_ids)).encode()).hexdigest()


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
        "schema_version": "phase15-review-v1",
        "case_count": REVIEW_CASE_COUNT,
        "case_ids_sha256": _case_id_digest(case.case_id for case in case_list),
        "holdout_case_count": 0,
        "language_distribution": counts("language"),
        "pipeline_stage_distribution": counts("pipeline_stage"),
        "legal_category_distribution": counts("legal_category"),
        "answerability_distribution": counts("answerability"),
        "severity_distribution": counts("severity"),
        "ai_preclassification_cases": sum(case.ai_suggestion is not None for case in case_list),
        "provenance": "PHASE15_DEV",
    }


def prepare_review_packet(
    cases: Iterable[ReviewCase], packet_path: Path, manifest_path: Path
) -> dict[str, Any]:
    case_list = tuple(cases)
    manifest = build_review_manifest(case_list)
    packet = {
        "schema_version": "phase15-review-v1",
        "case_ids": [case.case_id for case in case_list],
        "cases": [case.model_dump(mode="json") for case in case_list],
    }
    write_json_atomic(packet_path, packet)
    write_json_atomic(manifest_path, manifest)
    return manifest


class ReviewStore:
    """Atomic progress store keyed by immutable packet case IDs."""

    def __init__(self, packet_path: Path, progress_path: Path) -> None:
        self.packet_path = packet_path
        self.progress_path = progress_path

    def cases(self) -> tuple[ReviewCase, ...]:
        payload = json.loads(self.packet_path.read_text(encoding="utf-8"))
        cases = tuple(ReviewCase.model_validate(item) for item in payload.get("cases", ()))
        if len(cases) != REVIEW_CASE_COUNT or len({case.case_id for case in cases}) != REVIEW_CASE_COUNT:
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

    def save_decision(self, decision: ReviewDecision) -> None:
        valid_ids = {case.case_id for case in self.cases()}
        if decision.case_id not in valid_ids:
            raise ValueError(f"unknown immutable review case ID: {decision.case_id}")
        decisions = self._decisions()
        decisions[decision.case_id] = decision
        write_json_atomic(
            self.progress_path,
            {
                "schema_version": "phase15-review-progress-v1",
                "decisions": {
                    key: value.model_dump(mode="json") for key, value in sorted(decisions.items())
                },
            },
        )

    def reviewed_count(self) -> int:
        return len(self._decisions())

    def status(self) -> dict[str, Any]:
        reviewed = self.reviewed_count()
        return {
            "reviewed": reviewed,
            "total": REVIEW_CASE_COUNT,
            "remaining": REVIEW_CASE_COUNT - reviewed,
            "progress": f"{reviewed} / {REVIEW_CASE_COUNT}",
            "packet_path": self.packet_path.as_posix(),
            "progress_path": self.progress_path.as_posix(),
            "finalize_ready": reviewed >= MINIMUM_FINAL_REVIEWS,
        }

    def next_unreviewed(self) -> ReviewCase | None:
        decisions = self._decisions()
        return next((case for case in self.cases() if case.case_id not in decisions), None)

    def require_finalize_ready(self) -> None:
        reviewed = self.reviewed_count()
        if reviewed < MINIMUM_FINAL_REVIEWS:
            raise RuntimeError(
                f"phase15 finalize requires >= {MINIMUM_FINAL_REVIEWS} unique human review decisions; got {reviewed}"
            )


def default_review_paths(root: Path = Path(".")) -> tuple[Path, Path, Path]:
    private_root = root / REVIEW_ROOT
    return (
        private_root / PACKET_FILENAME,
        private_root / PROGRESS_FILENAME,
        root / "data/manifests/evaluation/phase15_review_manifest.json",
    )
