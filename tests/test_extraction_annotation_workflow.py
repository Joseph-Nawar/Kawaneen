from kawaneen.extraction.annotation import AnnotationUpdate
from kawaneen.extraction.orchestration import annotation_progress, next_dev_annotation


def test_dev_progress_reports_only_the_current_unreviewed_dev_pack() -> None:
    progress = annotation_progress("dev")
    assert progress == {
        "split": "dev",
        "total": 80,
        "reviewed": 0,
        "human_verified": 0,
        "remaining": 0,
        "invalid": 0,
    }


def test_next_annotation_is_dev_only_and_exposes_candidates() -> None:
    payload = next_dev_annotation()
    assert payload["split"] == "dev"
    assert payload["annotation_status"] == "unreviewed"
    assert "canonical_text" in payload
    assert "candidate_registry" in payload
    assert "holdout" not in payload


def test_annotation_update_is_strict_and_starts_as_reviewable_semantics() -> None:
    update = AnnotationUpdate(
        human_annotations={"schema_version": "phase11-proposal-v1"},
        annotation_status="reviewed",
        human_verified=True,
    )
    assert update.human_verified is True
