from __future__ import annotations

from pathlib import Path

import pytest

import kawaneen.evaluation.orchestrator as evaluation_orchestrator
from kawaneen.evaluation.orchestrator import evaluation_plan, freeze_evaluation, run_build_draft


def test_evaluation_plan_is_text_free_and_clean_freeze_requires_a_draft(monkeypatch) -> None:
    plan = evaluation_plan()
    assert plan["private_root"] == "artifacts/private/phase6_evaluation"
    assert "query_text" not in str(plan)
    monkeypatch.setattr(
        evaluation_orchestrator,
        "_active_draft_paths",
        lambda: (Path("/tmp/missing-items.jsonl"), Path(), Path(), Path(), False),
    )
    result = freeze_evaluation()
    assert result["status"] == "blocked_missing_draft"


@pytest.mark.private_artifact
def test_build_draft_writes_private_outputs_and_sanitized_summary(tmp_path: Path) -> None:
    result = run_build_draft(private_root=tmp_path / "private", tracked_root=tmp_path / "tracked")
    assert result["item_count"] == 240
    assert (tmp_path / "private" / "draft" / "selected_and_variants.jsonl").is_file()
    assert (tmp_path / "tracked" / "phase6_draft_summary.json").is_file()
    assert "query_text" not in (tmp_path / "tracked" / "phase6_draft_summary.json").read_text()
