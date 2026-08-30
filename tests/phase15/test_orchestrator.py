from __future__ import annotations

import json
from pathlib import Path

from kawaneen.phase15.orchestrator import (
    phase15_freeze,
    phase15_model_lock,
    phase15_plan,
    phase15_review_prepare,
    write_phase15_status_artifacts,
)


def test_plan_and_freeze_write_only_governance_artifacts(tmp_path: Path) -> None:
    for phase in (*range(3, 12), 14):
        path = tmp_path / "data" / "evaluation" / f"phase{phase}_frozen.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    plan = phase15_plan(tmp_path)
    assert plan["seed"] == 20260826
    result = phase15_freeze(tmp_path)
    assert result["registry_entries"] == 10
    assert (tmp_path / "data/manifests/evaluation/phase15_experiment_plan.json").is_file()
    assert (tmp_path / "data/manifests/evaluation/phase15_evidence_registry.json").is_file()
    assert not (tmp_path / "data/evaluation/phase15_research_questions.json").exists()
    stored = json.loads(
        (tmp_path / "data/manifests/evaluation/phase15_experiment_plan.json").read_text()
    )
    assert stored["base_sha"] == "03f58284426c84c6c813be2b1e1bbbbbfd1c9a2d"


def test_status_artifacts_are_aggregate_and_do_not_claim_results(tmp_path: Path) -> None:
    result = write_phase15_status_artifacts(tmp_path)
    assert len(result) == 9
    assert (tmp_path / "data/manifests/evaluation/phase15_dialect_manifest.json").is_file()
    assert (tmp_path / "data/manifests/evaluation/phase15_generator_subset_manifest.json").is_file()
    assert not (tmp_path / "data/evaluation/phase15_dialect_manifest.json").exists()
    for path in result:
        payload = json.loads(Path(path).read_text())
        assert payload["status"] == "NOT_RUN"
        assert payload["provenance"] == "PHASE15_DEV"


def test_fallback_lock_uses_permissive_arabic_candidate(tmp_path: Path) -> None:
    payload = phase15_model_lock(tmp_path)
    assert payload["allam_status"].startswith("BLOCKED_BEFORE_SCORING")
    lock = json.loads((tmp_path / "data/manifests/evaluation/phase15_model_lock.json").read_text())
    assert lock["fallback_preregistered_before_results"]["model_id"] == (
        "abdelrahman-alkhodary/qwen2.5-1.5b-arabic-instruct"
    )
    assert lock["fallback_preregistered_before_results"]["revision"] == (
        "06d27020b3ac3d9058b7eebded9754c8e10fa6bd"
    )
    assert lock["fallback_preregistered_before_results"]["license"] == "apache-2.0"


def test_review_prepare_regenerates_only_when_progress_is_zero(tmp_path: Path) -> None:
    from kawaneen.phase15.contracts import ReviewCase

    cases = [
        ReviewCase(
            case_id=f"case-{i}",
            language="ar",
            pipeline_stage="retrieval",
            legal_category="regulatory",
            answerability="answerable",
            severity="medium",
        ).model_dump(mode="json")
        for i in range(120)
    ]
    candidate_path = tmp_path / "artifacts/private/phase15_evaluation/review_candidates.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    first = phase15_review_prepare(tmp_path)
    assert first["status"] == "prepared"
    cases[0]["pipeline_stage"] = "generation"
    candidate_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    regenerated = phase15_review_prepare(tmp_path)
    assert regenerated["status"] == "regenerated"
    packet = json.loads(Path(regenerated["packet"]).read_text(encoding="utf-8"))
    assert packet["cases"][0]["pipeline_stage"] == "generation"


def test_review_prepare_preserves_frozen_case_id_hash(tmp_path: Path) -> None:
    from kawaneen.phase15.contracts import ReviewCase

    cases = [
        ReviewCase(
            case_id=f"case-{i}",
            language="ar",
            pipeline_stage="retrieval",
            legal_category="regulatory",
            answerability="answerable",
            severity="medium",
        ).model_dump(mode="json")
        for i in range(120)
    ]
    candidate_path = tmp_path / "artifacts/private/phase15_evaluation/review_candidates.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    first = phase15_review_prepare(tmp_path)
    first_manifest = json.loads(Path(first["manifest"]).read_text())
    second = phase15_review_prepare(tmp_path)
    second_manifest = json.loads(Path(second["manifest"]).read_text())
    assert second_manifest["case_ids_sha256"] == first_manifest["case_ids_sha256"]
