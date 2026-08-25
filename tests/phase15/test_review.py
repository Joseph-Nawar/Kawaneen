from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawaneen.phase15.contracts import ErrorCategory, ReviewCase, ReviewDecision
from kawaneen.phase15.review import (
    ReviewStore,
    build_review_manifest,
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


def test_atomic_review_progress_resumes_and_deduplicates(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    prepare_review_packet(_cases(), packet_path, tmp_path / "manifest.json")
    store = ReviewStore(packet_path, tmp_path / "progress.json")
    decision = ReviewDecision(case_id="case-1", primary=ErrorCategory.OCR_FAILURE)
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
