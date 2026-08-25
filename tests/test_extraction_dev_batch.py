from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.annotation import AnnotationRecord
from kawaneen.extraction.artifacts import write_private_json, write_text_free_json
from kawaneen.extraction.candidates import build_candidate_registry
from kawaneen.extraction.contracts import ProposedRule, ProposedSpan, SemanticProposal
from kawaneen.extraction.orchestration import (
    PHASE11_SELECTION_FINGERPRINT,
    SEMANTIC_RELEASE_PATH,
    export_dev_annotation_batch,
    export_holdout_annotation_batch,
    freeze_holdout_annotation_release,
    freeze_stage_b2_configuration,
    import_adjudicated_holdout,
    import_reviewed_dev,
    import_reviewed_holdout,
)


def make_record(record_id: str, split: str = "dev") -> AnnotationRecord:
    text = "يجب تقديم الطلب خلال ٣٠ يوماً." if record_id.endswith("1") else "نص وصفي."
    return AnnotationRecord(
        canonical_unit_id=record_id,
        document_id=f"document-{record_id}",
        canonical_text=text,
        source_provenance=SourceProvenance(
            source_id="saudi-moj-derived",
            source_version="synthetic",
            source_path="private",
            source_row=1,
            source_field="text",
        ),
        source_fingerprint="f" * 64,
        split=split,
        candidate_registry=build_candidate_registry(
            text,
            canonical_unit_id=record_id,
            document_id=f"document-{record_id}",
        ),
    )


def make_fixture(tmp_path: Path) -> tuple[Path, Path, list[AnnotationRecord]]:
    annotation_root = tmp_path / "annotations"
    manifest_path = tmp_path / "selection.json"
    dev_records = [make_record(f"dev-{index}") for index in range(1, 81)]
    dev_records[0] = dev_records[0].model_copy(
        update={"annotation_status": "reviewed", "human_verified": True}
    )
    holdout = make_record("holdout-1", split="holdout")
    rows = [
        {
            "canonical_unit_id": record.canonical_unit_id,
            "document_id": record.document_id,
            "source_id": "saudi-moj-derived",
            "source_fingerprint": "f" * 64,
            "split": record.split,
            "smoke": False,
            "strata": [],
        }
        for record in (*dev_records, holdout)
    ]
    write_text_free_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifact_type": "phase11_annotation_selection",
            "selection_fingerprint": "expected-selection-fingerprint",
            "rows": rows,
        },
    )
    for record in (*dev_records, holdout):
        filename = hashlib.sha256(record.canonical_unit_id.encode()).hexdigest() + ".json"
        write_private_json(annotation_root / filename, record.model_dump(mode="json"))
    return annotation_root, manifest_path, dev_records


def reviewed_batch(records: list[AnnotationRecord], fingerprint: str) -> dict[str, object]:
    payload_records: list[dict[str, object]] = []
    for record in records:
        payload = record.model_dump(mode="json")
        payload.update(
            {
                "human_annotations": {"schema_version": "phase11-proposal-v1"},
                "annotation_status": "independent_ai_review",
                "annotation_provenance": "independent_ai_review",
                "human_verified": False,
            }
        )
        payload_records.append(payload)
    return {
        "schema_version": "phase11-dev-annotation-batch-v1",
        "artifact_type": "phase11_dev_annotation_batch",
        "selection_fingerprint": fingerprint,
        "annotation_provenance": "independent_ai_review",
        "records": payload_records,
    }


def test_export_contains_only_dev_records_and_preserves_originals(tmp_path: Path) -> None:
    annotation_root, manifest_path, dev_records = make_fixture(tmp_path)
    before = {
        record.canonical_unit_id: (
            annotation_root
            / f"{hashlib.sha256(record.canonical_unit_id.encode()).hexdigest()}.json"
        ).read_bytes()
        for record in dev_records
    }
    output_path = tmp_path / "review" / "batch.json"

    result = export_dev_annotation_batch(
        annotation_root=annotation_root,
        selection_manifest_path=manifest_path,
        output_path=output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["record_count"] == 80
    assert payload["annotation_contract"]["schema_version"] == "phase11-proposal-v1"
    assert payload["annotation_contract"]["allowed_modalities"] == [
        "obligation",
        "prohibition",
        "permission",
    ]
    assert len(payload["records"]) == 80
    assert {item["split"] for item in payload["records"]} == {"dev"}
    assert all(item["human_verified"] is False for item in payload["records"])
    assert all(item["human_annotations"] is None for item in payload["records"])
    assert all(item["candidate_registry"]["candidates"] for item in payload["records"][:1])
    assert all(
        (
            annotation_root
            / f"{hashlib.sha256(record.canonical_unit_id.encode()).hexdigest()}.json"
        ).read_bytes()
        == before[record.canonical_unit_id]
        for record in dev_records
    )


def test_export_holdout_batch_is_source_only_and_rebuilds_candidates_v3(tmp_path: Path) -> None:
    annotation_root = tmp_path / "annotations"
    manifest_path = tmp_path / "selection.json"
    records = [make_record(f"holdout-{index}", split="holdout") for index in range(40)]
    rows = [
        {
            "canonical_unit_id": record.canonical_unit_id,
            "document_id": record.document_id,
            "source_id": "saudi-moj-derived",
            "source_fingerprint": "f" * 64,
            "split": "holdout",
            "smoke": False,
            "strata": [],
        }
        for record in records
    ]
    write_text_free_json(
        manifest_path,
        {
            "schema_version": 2,
            "artifact_type": "phase11_annotation_selection",
            "selection_version": "phase11-selection-v2",
            "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
            "rows": rows,
        },
    )
    for record in records:
        write_private_json(
            annotation_root
            / f"{hashlib.sha256(record.canonical_unit_id.encode()).hexdigest()}.json",
            record.model_dump(mode="json"),
        )
    output_path = tmp_path / "review" / "holdout.json"

    result = export_holdout_annotation_batch(
        annotation_root=annotation_root,
        selection_manifest_path=manifest_path,
        output_path=output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["record_count"] == 40
    assert result["dev_records"] == 0
    assert result["holdout_records"] == 40
    assert payload["split"] == "holdout"
    assert payload["candidate_registry_version"] == "phase11-candidates-v3"
    assert all(item["split"] == "holdout" for item in payload["records"])
    assert all(item["annotation_status"] == "unreviewed" for item in payload["records"])
    assert all(item["annotation_provenance"] == "unreviewed" for item in payload["records"])
    assert all(item["human_annotations"] is None for item in payload["records"])
    assert all(item["human_verified"] is False for item in payload["records"])
    assert all("result" not in item and "raw_response" not in item for item in payload["records"])


@pytest.mark.private_artifact
def test_freeze_stage_b2_configuration_writes_text_free_selection_manifest(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "b2-config.json"
    evaluation_path = tmp_path / "b2-evaluation.json"
    comparison_path = tmp_path / "selection.json"
    output_path = tmp_path / "frozen.json"
    config = {
        "stage": "hybrid-qwen-v1-stage-b2",
        "split": "dev",
        "record_count": 80,
        "lifecycle_state": "complete",
        "holdout_access": 0,
        "template_version": "phase11-hybrid-qwen-stage-b2-prompt-v1",
        "template_sha256": "t" * 64,
        "schema_sha256": "s" * 64,
        "candidate_policy_version": "phase11-candidates-v3",
        "candidate_policy_hash": "dcc40496967242ee1cba99576bebce9eca3520d639f4ae25a6bb2fe0797cd675",
        "hf_model": "Qwen/Qwen3-4B-Instruct-2507",
        "hf_revision": "h" * 40,
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "ollama_tag": "qwen3:4b-instruct-2507-q4_K_M",
        "ollama_digest": "sha256:" + "d" * 64,
        "runtime_settings": {"max_output_tokens": 1024, "temperature": 0, "automatic_retries": 0},
    }
    write_text_free_json(config_path, config)
    write_text_free_json(evaluation_path, {"artifact_type": "evaluation"})
    comparison = {
        "selection": "SELECT_STAGE_B2_EXPERIMENTAL",
        "holdout_recommendation": "FREEZE_B2_AND_RUN_HOLDOUT_ONCE",
        "record_count": 80,
        "reference_status": "AI-reviewed/adjudicated; not human gold",
        "hashes": {
            "b2_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "b2_evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
            "b2_result_set_sha256": "r" * 64,
        },
        "selection_reason": "bounded synthetic test rationale",
    }
    write_text_free_json(comparison_path, comparison)
    result = freeze_stage_b2_configuration(
        comparison_path=comparison_path,
        b2_config_path=config_path,
        b2_evaluation_path=evaluation_path,
        semantic_release_path=SEMANTIC_RELEASE_PATH,
        output_path=output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["status"] == "DEV_FROZEN_PENDING_PROTECTED_HOLDOUT"
    assert payload["selection"] == "SELECT_STAGE_B2_EXPERIMENTAL"
    assert payload["records"] == 80
    assert payload["holdout_model_inference"] == 0
    assert payload["human_gold"] is False


def test_import_reviewed_holdout_preserves_ai_provenance_and_validates_spans(
    tmp_path: Path,
) -> None:
    annotation_root = tmp_path / "annotations"
    manifest_path = tmp_path / "selection.json"
    records = [make_record(f"holdout-{index}", split="holdout") for index in range(40)]
    rows = [
        {
            "canonical_unit_id": record.canonical_unit_id,
            "document_id": record.document_id,
            "source_id": "saudi-moj-derived",
            "source_fingerprint": "f" * 64,
            "split": "holdout",
            "smoke": False,
            "strata": [],
        }
        for record in records
    ]
    write_text_free_json(
        manifest_path,
        {
            "schema_version": 2,
            "artifact_type": "phase11_annotation_selection",
            "selection_version": "phase11-selection-v2",
            "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
            "rows": rows,
        },
    )
    payload_records: list[dict[str, object]] = []
    for record in records:
        write_private_json(
            annotation_root
            / f"{hashlib.sha256(record.canonical_unit_id.encode()).hexdigest()}.json",
            record.model_dump(mode="json"),
        )
        payload = record.model_dump(mode="json")
        payload.update(
            {
                "human_annotations": {"schema_version": "phase11-proposal-v1"},
                "annotation_status": "independent_ai_review",
                "annotation_provenance": "independent_ai_review",
                "human_verified": False,
            }
        )
        payload_records.append(payload)
    reviewed_path = tmp_path / "holdout-reviewed.json"
    write_private_json(
        reviewed_path,
        {
            "schema_version": "phase11-holdout-annotation-batch-v1",
            "artifact_type": "phase11_holdout_annotation_batch",
            "split": "holdout",
            "selection_version": "phase11-selection-v2",
            "candidate_registry_version": "phase11-candidates-v3",
            "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
            "annotation_provenance": "independent_ai_review",
            "records": payload_records,
        },
    )

    result = import_reviewed_holdout(
        reviewed_path,
        annotation_root=annotation_root,
        selection_manifest_path=manifest_path,
    )
    stored = json.loads(
        (annotation_root / f"{hashlib.sha256(b'holdout-0').hexdigest()}.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["imported_records"] == 40
    assert stored["annotation_status"] == "independent_ai_review"
    assert stored["annotation_provenance"] == "independent_ai_review"
    assert stored["human_verified"] is False


def test_import_adjudication_promotes_only_the_excluded_exact_agreement(
    tmp_path: Path,
) -> None:
    annotation_root = tmp_path / "annotations"
    manifest_path = tmp_path / "selection.json"
    records = [make_record(f"holdout-{index}", split="holdout") for index in range(40)]
    rows = [
        {
            "canonical_unit_id": record.canonical_unit_id,
            "document_id": record.document_id,
            "source_id": "saudi-moj-derived",
            "source_fingerprint": "f" * 64,
            "split": "holdout",
            "smoke": False,
            "strata": [],
        }
        for record in records
    ]
    write_text_free_json(
        manifest_path,
        {
            "schema_version": 2,
            "artifact_type": "phase11_annotation_selection",
            "selection_version": "phase11-selection-v2",
            "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
            "corpus_fingerprint": "c" * 64,
            "rows": rows,
        },
    )
    first_records: list[dict[str, object]] = []
    second_records: list[dict[str, object]] = []
    for record in records:
        write_private_json(
            annotation_root
            / f"{hashlib.sha256(record.canonical_unit_id.encode()).hexdigest()}.json",
            record.model_dump(mode="json"),
        )
        first = record.model_dump(mode="json")
        first.update(
            {
                "human_annotations": {"schema_version": "phase11-proposal-v1"},
                "annotation_status": "independent_ai_review",
                "annotation_provenance": "independent_ai_review",
                "human_verified": False,
            }
        )
        second = dict(first)
        first_records.append(first)
        second_records.append(second)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_private_json(
        first_path,
        {
            "schema_version": "phase11-holdout-annotation-batch-v1",
            "artifact_type": "phase11_holdout_annotation_batch",
            "split": "holdout",
            "selection_version": "phase11-selection-v2",
            "candidate_registry_version": "phase11-candidates-v3",
            "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
            "annotation_provenance": "independent_ai_review",
            "records": first_records,
        },
    )
    write_private_json(
        second_path,
        {
            "schema_version": "phase11-holdout-second-ai-review-v1",
            "artifact_type": "phase11_holdout_second_ai_review",
            "split": "holdout",
            "selection_version": "phase11-selection-v2",
            "candidate_registry_version": "phase11-candidates-v3",
            "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
            "second_review_provenance": "independent_ai_review_2",
            "records": second_records,
        },
    )
    adjudication_records = []
    for index, record in enumerate(first_records[:-1]):
        adjudication_records.append(
            {
                "record_id": record["canonical_unit_id"],
                "decision": "review_1" if index < 37 else "custom",
                "adjudication_provenance": "ai_adjudicated_after_independent_second_review",
                "final_annotation": record["human_annotations"],
                "human_verified": False,
            }
        )
    adjudication_path = tmp_path / "adjudication.json"
    write_private_json(
        adjudication_path,
        {
            "schema_version": "phase11-holdout-adjudication-v1",
            "artifact_type": "phase11_holdout_ai_review_adjudication",
            "split": "holdout",
            "record_count": 39,
            "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
            "review_1_sha256": hashlib.sha256(first_path.read_bytes()).hexdigest(),
            "review_2_sha256": hashlib.sha256(second_path.read_bytes()).hexdigest(),
            "source_disagreement_sha256": "d" * 64,
            "adjudication_provenance": "ai_adjudicated_after_independent_second_review",
            "human_verified": False,
            "records": adjudication_records,
        },
    )
    result = import_adjudicated_holdout(
        adjudication_path,
        disagreement_path=tmp_path / "disagreements.json",
        first_review_path=first_path,
        second_review_path=second_path,
        annotation_root=annotation_root,
        selection_manifest_path=manifest_path,
        expected_sha256=None,
        verify_review_hashes=False,
    )

    assert result["adjudicated_records"] == 39
    assert result["dual_ai_agreed_records"] == 1
    stored = [
        json.loads(
            (
                annotation_root
                / f"{hashlib.sha256(record.canonical_unit_id.encode()).hexdigest()}.json"
            ).read_text(encoding="utf-8")
        )
        for record in records
    ]
    assert sum(item["annotation_provenance"] == "dual_ai_agreed" for item in stored) == 1
    assert (
        sum(
            item["annotation_provenance"] == "ai_adjudicated_after_independent_second_review"
            for item in stored
        )
        == 39
    )
    assert all(item["human_verified"] is False for item in stored)


@pytest.mark.private_artifact
def test_holdout_release_freeze_is_text_free_and_counts_provenance(
    tmp_path: Path,
) -> None:
    annotation_root = tmp_path / "annotations"
    manifest_path = tmp_path / "selection.json"
    records = [
        make_record(f"holdout-{index}", split="holdout").model_copy(
            update={
                "annotation_status": "dual_ai_agreed" if index == 0 else "reviewed",
                "annotation_provenance": (
                    "dual_ai_agreed"
                    if index == 0
                    else "ai_adjudicated_after_independent_second_review"
                ),
                "human_annotations": SemanticProposal(schema_version="phase11-proposal-v1"),
            }
        )
        for index in range(40)
    ]
    rows = [
        {
            "canonical_unit_id": record.canonical_unit_id,
            "document_id": record.document_id,
            "source_id": "saudi-moj-derived",
            "source_fingerprint": "f" * 64,
            "split": "holdout",
            "smoke": False,
            "strata": [],
        }
        for record in records
    ]
    write_text_free_json(
        manifest_path,
        {
            "schema_version": 2,
            "artifact_type": "phase11_annotation_selection",
            "selection_version": "phase11-selection-v2",
            "selection_fingerprint": "expected-selection-fingerprint",
            "corpus_fingerprint": "c" * 64,
            "rows": rows,
        },
    )
    for record in records:
        write_private_json(
            annotation_root
            / f"{hashlib.sha256(record.canonical_unit_id.encode()).hexdigest()}.json",
            record.model_dump(mode="json"),
        )
    private = tmp_path / "release.json"
    tracked = tmp_path / "manifest.json"
    adjudication_path = tmp_path / "adjudication.json"
    write_private_json(adjudication_path, {"artifact_type": "synthetic-adjudication"})
    result = freeze_holdout_annotation_release(
        annotation_root=annotation_root,
        selection_manifest_path=manifest_path,
        private_output_path=private,
        tracked_output_path=tracked,
        adjudication_path=adjudication_path,
    )
    assert result["record_count"] == 40
    assert result["dual_ai_agreed"] == 1
    assert result["ai_adjudicated_after_independent_second_review"] == 39
    assert "canonical_text" not in json.loads(tracked.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="already frozen"):
        freeze_holdout_annotation_release(
            annotation_root=annotation_root,
            selection_manifest_path=manifest_path,
            private_output_path=private,
            tracked_output_path=tracked,
            adjudication_path=adjudication_path,
        )


def test_import_sets_ai_provenance_and_never_human_verified(tmp_path: Path) -> None:
    annotation_root, manifest_path, dev_records = make_fixture(tmp_path)
    reviewed_path = tmp_path / "reviewed.json"
    write_private_json(
        reviewed_path,
        reviewed_batch(dev_records, "expected-selection-fingerprint"),
    )

    result = import_reviewed_dev(
        reviewed_path,
        annotation_root=annotation_root,
        selection_manifest_path=manifest_path,
    )
    stored = json.loads(
        (annotation_root / f"{hashlib.sha256(b'dev-1').hexdigest()}.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["imported_records"] == 80
    assert stored["annotation_status"] == "independent_ai_review"
    assert stored["annotation_provenance"] == "independent_ai_review"
    assert stored["human_verified"] is False

    rejected = reviewed_batch(dev_records, "expected-selection-fingerprint")
    assert isinstance(rejected["records"][0], dict)
    rejected["records"][0]["human_verified"] = True
    rejected_path = tmp_path / "reviewed-human-gold.json"
    write_private_json(rejected_path, rejected)
    with pytest.raises(ValueError, match="human_verified"):
        import_reviewed_dev(
            rejected_path,
            annotation_root=annotation_root,
            selection_manifest_path=manifest_path,
        )


def test_import_allows_cross_role_span_reuse(tmp_path: Path) -> None:
    annotation_root, manifest_path, dev_records = make_fixture(tmp_path)
    reviewed = reviewed_batch(dev_records, "expected-selection-fingerprint")
    reviewed_record = reviewed["records"][0]
    assert isinstance(reviewed_record, dict)
    actor = ProposedSpan(text="الطلب")
    reviewed_record["human_annotations"] = SemanticProposal(
        schema_version="phase11-proposal-v1",
        regulated_entities=(actor,),
        rules=(
            ProposedRule(
                modality="obligation",
                actor=actor,
                action=ProposedSpan(text="تقديم"),
                deadline_refs=("T001",),
            ),
        ),
        deadline_refs=("T001",),
    ).model_dump(mode="json")
    reviewed_path = tmp_path / "reviewed-cross-role.json"
    write_private_json(reviewed_path, reviewed)

    result = import_reviewed_dev(
        reviewed_path,
        annotation_root=annotation_root,
        selection_manifest_path=manifest_path,
    )

    assert result["imported_records"] == 80


def test_import_rejects_fingerprint_missing_duplicate_and_holdout_ids(tmp_path: Path) -> None:
    annotation_root, manifest_path, dev_records = make_fixture(tmp_path)
    reviewed_path = tmp_path / "reviewed.json"
    payload = reviewed_batch(dev_records[:1], "wrong-fingerprint")
    write_private_json(reviewed_path, payload)
    with pytest.raises(ValueError, match="selection fingerprint"):
        import_reviewed_dev(
            reviewed_path,
            annotation_root=annotation_root,
            selection_manifest_path=manifest_path,
        )

    payload = reviewed_batch(dev_records[:1], "expected-selection-fingerprint")
    write_private_json(reviewed_path, payload)
    with pytest.raises(ValueError, match="missing DEV IDs"):
        import_reviewed_dev(
            reviewed_path,
            annotation_root=annotation_root,
            selection_manifest_path=manifest_path,
        )
    assert (
        import_reviewed_dev(
            reviewed_path,
            partial=True,
            annotation_root=annotation_root,
            selection_manifest_path=manifest_path,
        )["imported_records"]
        == 1
    )

    payload = reviewed_batch(dev_records, "expected-selection-fingerprint")
    payload["records"] = [payload["records"][0], payload["records"][0]]
    write_private_json(reviewed_path, payload)
    with pytest.raises(ValueError, match="duplicate"):
        import_reviewed_dev(
            reviewed_path,
            annotation_root=annotation_root,
            selection_manifest_path=manifest_path,
        )

    payload = reviewed_batch(
        [*dev_records, make_record("holdout-1", "holdout")], "expected-selection-fingerprint"
    )
    write_private_json(reviewed_path, payload)
    with pytest.raises(ValueError, match="not in DEV"):
        import_reviewed_dev(
            reviewed_path,
            annotation_root=annotation_root,
            selection_manifest_path=manifest_path,
        )


def test_import_rejects_invalid_semantic_span_without_writing(tmp_path: Path) -> None:
    annotation_root, manifest_path, dev_records = make_fixture(tmp_path)
    reviewed = reviewed_batch(dev_records, "expected-selection-fingerprint")
    reviewed_record = reviewed["records"][0]
    assert isinstance(reviewed_record, dict)
    reviewed_record["human_annotations"] = {
        "schema_version": "phase11-proposal-v1",
        "regulated_entities": [{"text": "not in source"}],
    }
    reviewed_path = tmp_path / "reviewed.json"
    write_private_json(reviewed_path, reviewed)

    with pytest.raises(ValueError, match="semantic span"):
        import_reviewed_dev(
            reviewed_path,
            annotation_root=annotation_root,
            selection_manifest_path=manifest_path,
        )


def test_import_rejects_invalid_candidate_reference(tmp_path: Path) -> None:
    annotation_root, manifest_path, dev_records = make_fixture(tmp_path)
    reviewed = reviewed_batch(dev_records, "expected-selection-fingerprint")
    reviewed_record = reviewed["records"][0]
    assert isinstance(reviewed_record, dict)
    reviewed_record["human_annotations"] = {
        "schema_version": "phase11-proposal-v1",
        "deadline_refs": ["T999"],
    }
    reviewed_path = tmp_path / "reviewed.json"
    write_private_json(reviewed_path, reviewed)

    with pytest.raises(ValueError, match="unknown candidate reference"):
        import_reviewed_dev(
            reviewed_path,
            annotation_root=annotation_root,
            selection_manifest_path=manifest_path,
        )


def test_future_annotation_states_are_additive_and_not_human_gold() -> None:
    record = make_record("dev-1").model_copy(
        update={
            "annotation_status": "dual_ai_disagreement",
            "annotation_provenance": "dual_ai_disagreement",
        }
    )
    assert record.human_verified is False
