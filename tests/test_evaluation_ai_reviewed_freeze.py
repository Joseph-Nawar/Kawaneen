from __future__ import annotations

import hashlib
import json
from pathlib import Path

import kawaneen.evaluation.orchestrator as evaluation_orchestrator
from kawaneen.evaluation.orchestrator import (
    evaluation_plan,
    evaluation_stats,
    freeze_ai_reviewed_release,
)

FINAL_ITEMS = Path(
    "artifacts/private/phase6_evaluation/final-candidate-v1/draft/selected_and_variants.jsonl"
)


def test_ai_reviewed_release_is_byte_preserving_and_not_human_gated(tmp_path: Path) -> None:
    source_bytes = FINAL_ITEMS.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()

    result = freeze_ai_reviewed_release(
        release_root=tmp_path / "private-release",
        manifest_path=tmp_path / "tracked" / "phase6_ai_reviewed_v1_manifest.json",
        report_path=tmp_path / "tracked" / "phase6_retrieval_eval_ai_reviewed_v1_report.json",
    )

    released = tmp_path / "private-release" / "draft" / "selected_and_variants.jsonl"
    assert released.read_bytes() == source_bytes
    assert hashlib.sha256(released.read_bytes()).hexdigest() == source_hash
    assert result["dataset_version"] == "phase6-retrieval-eval-ai-reviewed-v1"
    assert result["review_provenance"] == "independent_ai_source_review"
    assert result["human_verified"] is False
    assert result["formal_human_review_required"] is False

    manifest = json.loads(
        (tmp_path / "tracked" / "phase6_ai_reviewed_v1_manifest.json").read_text()
    )
    report = json.loads(
        (tmp_path / "tracked" / "phase6_retrieval_eval_ai_reviewed_v1_report.json").read_text()
    )
    assert manifest["item_count"] == 240
    assert manifest["base_intent_count"] == 200
    assert manifest["variant_count"] == 40
    assert manifest["human_verified"] is False
    assert report["limitation_statement"]
    assert "query_text" not in json.dumps(manifest, ensure_ascii=False)
    assert "gold_answer" not in json.dumps(report, ensure_ascii=False)


def test_ai_reviewed_release_rejects_mutated_existing_release(tmp_path: Path) -> None:
    release_root = tmp_path / "private-release"
    tracked_root = tmp_path / "tracked"
    freeze_ai_reviewed_release(
        release_root=release_root,
        manifest_path=tracked_root / "phase6_ai_reviewed_v1_manifest.json",
        report_path=tracked_root / "phase6_retrieval_eval_ai_reviewed_v1_report.json",
    )
    released = release_root / "draft" / "selected_and_variants.jsonl"
    released.write_bytes(released.read_bytes() + b"\n")

    try:
        freeze_ai_reviewed_release(
            release_root=release_root,
            manifest_path=tracked_root / "phase6_ai_reviewed_v1_manifest.json",
            report_path=tracked_root / "phase6_retrieval_eval_ai_reviewed_v1_report.json",
        )
    except ValueError as exc:
        assert "immutable" in str(exc)
    else:
        raise AssertionError("mutated AI-reviewed release was accepted")


def test_active_phase6_metadata_reports_ai_reviewed_release() -> None:
    plan = evaluation_plan()
    stats = evaluation_stats()
    assert plan["active_draft_version"] == "phase6-retrieval-eval-ai-reviewed-v1"
    assert plan["release_classification"] == "externally_ai_reviewed"
    assert stats["status"] == "frozen_ai_reviewed_engineering_release"
    assert stats["human_verified"] is False
    assert stats["formal_human_review_required"] is False


def test_stats_retains_pre_release_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_orchestrator,
        "AI_REVIEWED_MANIFEST_PATH",
        Path("/tmp/phase6-ai-reviewed-manifest-does-not-exist.json"),
    )
    stats = evaluation_stats()
    assert stats["status"] == "draft_pending_review"
    assert stats["item_count"] == 240


def test_stats_reports_missing_draft(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_orchestrator,
        "AI_REVIEWED_MANIFEST_PATH",
        Path("/tmp/phase6-ai-reviewed-manifest-does-not-exist.json"),
    )
    monkeypatch.setattr(
        evaluation_orchestrator,
        "_active_draft_paths",
        lambda: (Path("/tmp/missing-items.jsonl"), Path(), Path(), Path(), False),
    )
    assert evaluation_stats() == {"status": "missing_draft"}
