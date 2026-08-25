import pytest

import kawaneen.extraction.orchestration as extraction_orchestration
from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.annotation import AnnotationRecord, AnnotationUpdate
from kawaneen.extraction.contracts import CandidateRegistry
from kawaneen.extraction.orchestration import annotation_progress, next_dev_annotation


@pytest.mark.private_artifact
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


def test_next_annotation_is_dev_only_and_exposes_candidates(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    record = AnnotationRecord(
        canonical_unit_id="unit-1",
        document_id="document-1",
        canonical_text="Synthetic regulatory clause.",
        source_provenance=SourceProvenance(
            source_id="synthetic",
            source_version="v1",
            source_path="synthetic.json",
            source_row=1,
            source_field="text",
        ),
        source_fingerprint="a" * 64,
        split="dev",
        candidate_registry=CandidateRegistry(
            canonical_text="Synthetic regulatory clause.",
            canonical_unit_id="unit-1",
            document_id="document-1",
        ),
    )
    monkeypatch.setattr(extraction_orchestration, "_load_records", lambda _split: [record])

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
