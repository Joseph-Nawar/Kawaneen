import json
from pathlib import Path

import pytest

import kawaneen.retrieval.hybrid.orchestration as orchestration
from kawaneen.retrieval.hybrid.artifacts import is_text_free, write_json_atomic
from kawaneen.retrieval.hybrid.evaluation import select_reranker_pipeline
from kawaneen.retrieval.hybrid.finalization import validate_phase8_holdout_artifacts
from kawaneen.retrieval.hybrid.orchestration import validate_phase7_inputs


def test_tracked_artifact_text_free_check_rejects_query_and_passage_keys() -> None:
    assert is_text_free({"metrics": {"nDCG@10": 0.1}})
    assert not is_text_free({"query_text": "secret"})
    assert not is_text_free({"display_text": "secret"})


def test_atomic_json_write_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_json_atomic(path, {"metrics": {"nDCG@10": 0.2}})
    assert json.loads(path.read_text(encoding="utf-8"))["metrics"]["nDCG@10"] == 0.2


def test_phase7_selection_sha_is_required(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="selection SHA"):
        validate_phase7_inputs(selection_path=selection)


def test_phase8_status_does_not_load_model_or_full_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "provisional"}), encoding="utf-8")
    monkeypatch.setattr(orchestration, "PHASE8_SELECTION", selection)
    monkeypatch.setattr(orchestration, "PHASE8_PRIVATE", tmp_path / "private")
    monkeypatch.setattr(
        orchestration,
        "load_phase7_release",
        lambda: pytest.fail("status must not load the full corpus"),
    )
    result = orchestration.phase8_status()
    assert result["real_reranker_loaded"] is False


@pytest.mark.private_artifact
def test_completed_dev_reranker_artifacts_validate_without_loading_model() -> None:
    result = orchestration.validate_phase8_dev_reranker_artifacts()
    assert result["completed_query_count"] == 160
    assert result["invalid_query_count"] == 0
    assert result["duplicate_candidate_query_count"] == 0


@pytest.mark.private_artifact
def test_completed_holdout_artifacts_validate_without_model_execution() -> None:
    result = validate_phase8_holdout_artifacts()
    assert result["status"] == "validated"
    assert result["expected_query_count"] == 80
    assert result["completed_query_count"] == 80
    assert result["corrupt_query_count"] == 0
    assert result["duplicate_candidate_query_count"] == 0
    assert result["non_finite_score_query_count"] == 0
    assert result["config_hash_match"] is True


def test_reranker_selection_rule_applies_fixed_thresholds() -> None:
    rrf = {"nDCG@10": 0.1, "Recall@10": 0.2, "CompleteEvidenceRecall@10": 0.2}
    improved = {"nDCG@10": 0.1021, "Recall@10": 0.191, "CompleteEvidenceRecall@10": 0.191}
    rejected = {"nDCG@10": 0.1019, "Recall@10": 0.19, "CompleteEvidenceRecall@10": 0.19}

    assert select_reranker_pipeline(rrf, improved)["selected_pipeline"] == "rrf_reranked"
    assert select_reranker_pipeline(rrf, rejected)["selected_pipeline"] == "rrf"


def test_phase8_holdout_requires_explicit_permission_without_loading_holdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestration,
        "load_phase7_release",
        lambda **_: pytest.fail("holdout must not load without permission"),
    )
    with pytest.raises(PermissionError, match="--allow-holdout"):
        orchestration.phase8_holdout(allow_holdout=False, resume=True, device="cpu")
