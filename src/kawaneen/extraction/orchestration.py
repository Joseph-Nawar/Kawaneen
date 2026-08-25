"""Private Phase 11A extraction orchestration and protected CLI operations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal, cast

from kawaneen.extraction.annotation import (
    ANNOTATION_ROOT,
    PHASE11_ELIGIBILITY_POLICY_VERSION,
    PHASE11_SELECTION_VERSION,
    SELECTION_MANIFEST_PATH,
    AnnotationRecord,
    AnnotationUpdate,
    is_human_gold,
    prepare_annotation_pack,
    validate_annotation_record,
)
from kawaneen.extraction.artifacts import write_private_json, write_text_free_json
from kawaneen.extraction.candidates import CANDIDATE_REGISTRY_VERSION, build_candidate_registry
from kawaneen.extraction.checkpoints import ExtractionCheckpointStore
from kawaneen.extraction.contracts import SemanticProposal
from kawaneen.extraction.deterministic import run_deterministic
from kawaneen.extraction.hybrid_prompt import (
    HYBRID_PROMPT_TEMPLATE_VERSION,
    HYBRID_QWEN_HF_ID,
    HYBRID_QWEN_HF_REVISION,
    HYBRID_QWEN_MODEL,
    HYBRID_QWEN_OLLAMA_DIGEST,
    HYBRID_QWEN_TOKENIZER_REVISION,
    HYBRID_RUNTIME_SETTINGS,
    HYBRID_STAGE_B2_PROMPT_TEMPLATE_VERSION,
    hybrid_prompt_hash,
    hybrid_schema_hash,
)
from kawaneen.extraction.hybrid_runtime import run_hybrid_records
from kawaneen.extraction.provider import OllamaExtractionProvider
from kawaneen.extraction.readiness import (
    READINESS_MANIFEST_PATH,
    write_readiness_artifacts,
)

RESULT_ROOT = Path("artifacts/private/phase11_extraction/results")
REVIEW_ROOT = Path("artifacts/private/phase11_extraction/review")
DEV_BATCH_PATH = REVIEW_ROOT / "phase11_dev_annotation_batch_v1.json"
DEV_BATCH_V2_PATH = REVIEW_ROOT / "phase11_dev_annotation_batch_v2.json"
HOLDOUT_BATCH_PATH = REVIEW_ROOT / "phase11_holdout_annotation_batch_v1.json"
HOLDOUT_ADJUDICATION_PATH = REVIEW_ROOT / "phase11_holdout_ai_review_adjudication_v1.json"
HOLDOUT_FIRST_REVIEW_PATH = (
    REVIEW_ROOT / "phase11_holdout_annotation_batch_v1_independent_ai_review.json"
)
HOLDOUT_SECOND_REVIEW_PATH = REVIEW_ROOT / "phase11_holdout_second_ai_review_v1.json"
HOLDOUT_DISAGREEMENT_PATH = REVIEW_ROOT / "phase11_holdout_ai_review_disagreements_v1.json"
HOLDOUT_RELEASE_PATH = Path(
    "artifacts/private/phase11_extraction/releases/phase11_holdout_annotation_release_v1.json"
)
HOLDOUT_RELEASE_MANIFEST_PATH = Path(
    "data/manifests/extraction/phase11_holdout_annotation_release_v1.json"
)
CANDIDATE_AUDIT_V2_PATH = REVIEW_ROOT / "phase11_dev_candidate_audit_v2.json"
HYBRID_CHECKPOINT_ROOT = Path("artifacts/private/phase11_extraction/checkpoints/hybrid-qwen-v1/dev")
HYBRID_RESULT_ROOT = RESULT_ROOT / "hybrid-qwen-v1" / "dev"
HYBRID_CONFIG_PATH = Path("data/manifests/extraction/phase11_hybrid_dev_config_v1.json")
HYBRID_PRIVATE_METADATA_PATH = HYBRID_RESULT_ROOT / "experiment.json"
HYBRID_STAGE_B1_CHECKPOINT_ROOT = Path(
    "artifacts/private/phase11_extraction/checkpoints/hybrid-qwen-v1-stage-b1/dev"
)
HYBRID_STAGE_B1_RESULT_ROOT = RESULT_ROOT / "hybrid-qwen-v1-stage-b1" / "dev"
HYBRID_STAGE_B1_CONFIG_PATH = Path(
    "data/manifests/extraction/phase11_hybrid_stage_b1_config_v1.json"
)
HYBRID_STAGE_B1_PRIVATE_METADATA_PATH = HYBRID_STAGE_B1_RESULT_ROOT / "experiment.json"
HYBRID_STAGE_B1_CLEAN_CHECKPOINT_ROOT = Path(
    "artifacts/private/phase11_extraction/checkpoints/hybrid-qwen-v1-stage-b1-clean/dev"
)
HYBRID_STAGE_B1_CLEAN_RESULT_ROOT = RESULT_ROOT / "hybrid-qwen-v1-stage-b1-clean" / "dev"
HYBRID_STAGE_B1_CLEAN_CONFIG_PATH = Path(
    "data/manifests/extraction/phase11_hybrid_stage_b1_clean_config_v1.json"
)
HYBRID_STAGE_B1_CLEAN_PRIVATE_METADATA_PATH = HYBRID_STAGE_B1_CLEAN_RESULT_ROOT / "experiment.json"
HYBRID_STAGE_B2_CLEAN_CHECKPOINT_ROOT = Path(
    "artifacts/private/phase11_extraction/checkpoints/hybrid-qwen-v1-stage-b2-clean/dev"
)
HYBRID_STAGE_B2_CLEAN_RESULT_ROOT = RESULT_ROOT / "hybrid-qwen-v1-stage-b2-clean" / "dev"
HYBRID_STAGE_B2_CLEAN_CONFIG_PATH = Path(
    "data/manifests/extraction/phase11_hybrid_stage_b2_clean_config_v1.json"
)
HYBRID_STAGE_B2_CLEAN_PRIVATE_METADATA_PATH = HYBRID_STAGE_B2_CLEAN_RESULT_ROOT / "experiment.json"
HYBRID_STAGE_B2_HOLDOUT_CHECKPOINT_ROOT = Path(
    "artifacts/private/phase11_extraction/checkpoints/hybrid-qwen-v1-stage-b2-clean/holdout"
)
HYBRID_STAGE_B2_HOLDOUT_RESULT_ROOT = RESULT_ROOT / "hybrid-qwen-v1-stage-b2-clean" / "holdout"
HYBRID_STAGE_B2_HOLDOUT_CONFIG_PATH = Path(
    "data/manifests/extraction/phase11_hybrid_stage_b2_clean_holdout_config_v1.json"
)
HYBRID_STAGE_B2_HOLDOUT_PRIVATE_METADATA_PATH = (
    HYBRID_STAGE_B2_HOLDOUT_RESULT_ROOT / "experiment.json"
)
B2_SELECTION_ARTIFACT_PATH = Path("data/evaluation/phase11_stage_b1_vs_b2_selection_v1.json")
B2_SELECTED_CONFIG_PATH = Path(
    "data/manifests/extraction/phase11_hybrid_stage_b2_clean_config_v1.json"
)
B2_EVALUATION_PATH = Path(
    "artifacts/private/phase11_extraction/evaluation/"
    "hybrid-qwen-v1-stage-b2-clean/dev/evaluation_report_private.json"
)
PHASE11_SELECTED_CONFIGURATION_PATH = Path(
    "data/manifests/extraction/phase11_selected_configuration_v1.json"
)
SEMANTIC_RELEASE_PATH = Path(
    "artifacts/private/phase11_extraction/releases/phase11_dev_annotation_release_v1.json"
)
CANDIDATE_COMPATIBLE_RELEASE_PATH = Path(
    "artifacts/private/phase11_extraction/releases/phase11_dev_candidate_compatible_release_v1.json"
)
PHASE11_SEMANTIC_RELEASE_FINGERPRINT = (
    "a48be5b6cacb4b0a1d2a45c82d4f4b2b12ec80c725447d9ff20d2e8355c80f04"
)
PHASE11_SEMANTIC_RELEASE_SHA256 = "9e0f60baa9b19c7841c729643f257bb5be8a9f993e51b2d95660909421af53c1"
PHASE11_CANDIDATE_POLICY_HASH = "dcc40496967242ee1cba99576bebce9eca3520d639f4ae25a6bb2fe0797cd675"
PHASE11_CANDIDATE_COMPATIBLE_FINGERPRINT = (
    "b06fd5088e0e97250e6951064087a2c3a3a3f930213e73c7e10d7a996ef74e40"
)
PHASE11_SELECTION_FINGERPRINT = "98a698a82ac8d444ad35cb037bb7fadb5e1c4ca1c1d63e5cee13a9e273107f05"
PHASE11_HOLDOUT_ADJUDICATION_SHA256 = (
    "61fb85de41036b214ff776790dd9b696cbd48b83e50b590c66ad5e7293ad1ef0"
)
PHASE11_HOLDOUT_FIRST_REVIEW_SHA256 = (
    "4854bb22ee653e062f3796ede920c2693771be00d6ca971a40c40063ea5a0291"
)
PHASE11_HOLDOUT_SECOND_REVIEW_SHA256 = (
    "09d3416feee8ac80744c4f2ce9c92eac7590123fa718a94dcfc9c1e8e0e039a4"
)
PHASE11_HOLDOUT_DISAGREEMENT_SHA256 = (
    "8dcf33cb4c189399082fabf885d775846959333d075fe7ebf8fca82a99b4de12"
)


def _private_record_path(record_id: str, annotation_root: Path) -> Path:
    return annotation_root / f"{hashlib.sha256(record_id.encode('utf-8')).hexdigest()}.json"


def _load_records(
    split: Literal["dev", "holdout"],
    *,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
) -> list[AnnotationRecord]:
    if not annotation_root.is_dir():
        raise ValueError("annotation pack is unavailable; run extraction prepare-annotations first")
    if not selection_manifest_path.is_file():
        raise ValueError("annotation selection manifest is unavailable")
    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    rows = cast(list[dict[str, object]], selection_payload["rows"])
    selected_ids = [str(row["canonical_unit_id"]) for row in rows if str(row["split"]) == split]
    records: list[AnnotationRecord] = []
    for record_id in sorted(selected_ids):
        path = _private_record_path(record_id, annotation_root)
        if not path.is_file():
            raise ValueError(
                f"annotation record is missing for selected DEV/HOLDOUT unit: {record_id}"
            )
        record = AnnotationRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if record.canonical_unit_id != record_id or record.split != split:
            raise ValueError(f"annotation record does not match selection manifest: {record_id}")
        records.append(record)
    if not records:
        raise ValueError(f"annotation pack contains no {split} records")
    return records


def _guard_split(split: str, allow_holdout: bool) -> None:
    if split not in {"dev", "holdout", "smoke"}:
        raise ValueError("split must be dev, smoke, or holdout")
    if split == "holdout" and not allow_holdout:
        raise ValueError("HOLDOUT is sealed; pass --allow-holdout to use the protected command")


def extraction_status() -> dict[str, object]:
    selection_exists = SELECTION_MANIFEST_PATH.is_file()
    readiness_exists = READINESS_MANIFEST_PATH.is_file()
    return {
        "selection_manifest": selection_exists,
        "readiness_manifest": readiness_exists,
        "annotation_pack": ANNOTATION_ROOT.is_dir(),
        "model_loaded": False,
        "hybrid_model_calls": 0,
        "holdout_sealed": True,
        "holdout_evaluation_performed": False,
    }


def prepare_annotations() -> dict[str, object]:
    pack = prepare_annotation_pack()
    readiness = write_readiness_artifacts(pack)
    return {key: value for key, value in readiness.items() if key != "records"}


def validate_annotations(split: str, *, allow_holdout: bool = False) -> dict[str, object]:
    _guard_split(split, allow_holdout)
    records = _load_records("dev" if split == "smoke" else cast(Literal["dev", "holdout"], split))
    selected = [
        record
        for record in records
        if record.split == ("dev" if split == "smoke" else split)
        and (split != "smoke" or record.smoke)
    ]
    ids = {record.canonical_unit_id for record in selected}
    diagnostics = [
        error for record in selected for error in validate_annotation_record(record, ids)
    ]
    return {
        "split": split,
        "record_count": len(selected),
        "valid": not diagnostics,
        "diagnostics": diagnostics,
        "human_gold_records": sum(is_human_gold(record) for record in selected),
    }


def run_deterministic_split(split: str, *, allow_holdout: bool = False) -> dict[str, object]:
    _guard_split(split, allow_holdout)
    records = _load_records("dev" if split == "smoke" else cast(Literal["dev", "holdout"], split))
    selected = [
        record
        for record in records
        if record.split == ("dev" if split == "smoke" else split)
        and (split != "smoke" or record.smoke)
    ]
    selection_payload = json.loads(SELECTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    destination = RESULT_ROOT / "deterministic-v2" / split
    for record in selected:
        result = run_deterministic(
            record.canonical_text,
            canonical_unit_id=record.canonical_unit_id,
            document_id=record.document_id,
            source_provenance=record.source_provenance,
        )
        filename = hashlib.sha256(record.canonical_unit_id.encode("utf-8")).hexdigest() + ".json"
        write_private_json(
            destination / filename,
            {
                "artifact_type": "phase11_extraction_result",
                "lifecycle_state": "complete",
                "record_id": record.canonical_unit_id,
                "extractor": "deterministic-v1",
                "selection_version": selection_payload["selection_version"],
                "selection_fingerprint": selection_payload["selection_fingerprint"],
                "candidate_registry_version": CANDIDATE_REGISTRY_VERSION,
                "result": result.model_dump(mode="json"),
            },
        )
    return {
        "split": split,
        "extractor": "deterministic-v1",
        "selection_version": selection_payload["selection_version"],
        "record_count": len(selected),
        "model_calls": 0,
    }


def _load_holdout_source_records_for_inference(
    *,
    batch_path: Path = HOLDOUT_BATCH_PATH,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
) -> list[AnnotationRecord]:
    """Load only the source-only HOLDOUT batch for prediction generation."""

    payload_value = json.loads(batch_path.read_text(encoding="utf-8"))
    if not isinstance(payload_value, dict):
        raise ValueError("HOLDOUT source batch must be a JSON object")
    payload = cast(dict[str, object], payload_value)
    if (
        payload.get("schema_version") != "phase11-holdout-annotation-batch-v1"
        or payload.get("split") != "holdout"
        or payload.get("candidate_registry_version") != CANDIDATE_REGISTRY_VERSION
    ):
        raise ValueError("HOLDOUT inference source batch is not the locked candidates-v3 batch")
    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if payload.get("selection_fingerprint") != selection_payload.get("selection_fingerprint"):
        raise ValueError("HOLDOUT source batch selection fingerprint does not match selection")
    rows = cast(list[dict[str, object]], selection_payload["rows"])
    expected_ids = {str(row["canonical_unit_id"]) for row in rows if str(row["split"]) == "holdout"}
    raw_records_value = payload.get("records")
    if not isinstance(raw_records_value, list):
        raise ValueError("HOLDOUT source batch records must be a list")
    raw_records: list[dict[str, object]] = []
    for item in cast(list[object], raw_records_value):
        if isinstance(item, dict):
            raw_records.append(cast(dict[str, object], item))
    if {str(item.get("canonical_unit_id")) for item in raw_records} != expected_ids:
        raise ValueError("HOLDOUT source batch IDs do not match the protected selection")
    records: list[AnnotationRecord] = []
    for raw in raw_records:
        if raw.get("human_annotations") is not None or raw.get("human_verified") is True:
            raise ValueError("HOLDOUT inference source batch contains reference annotations")
        record = AnnotationRecord.model_validate(
            {
                **raw,
                "annotation_status": "unreviewed",
                "annotation_provenance": "unreviewed",
                "human_annotations": None,
                "human_verified": False,
            }
        )
        expected_registry = build_candidate_registry(
            record.canonical_text,
            canonical_unit_id=record.canonical_unit_id,
            document_id=record.document_id,
        )
        if record.candidate_registry != expected_registry:
            raise ValueError(
                f"HOLDOUT source candidate registry mismatch: {record.canonical_unit_id}"
            )
        records.append(record)
    return sorted(records, key=lambda item: item.canonical_unit_id)


def _candidate_collection_fingerprint(records: list[AnnotationRecord]) -> str:
    rows: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.canonical_unit_id):
        encoded = json.dumps(
            record.candidate_registry.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        rows.append(
            {
                "canonical_unit_id": record.canonical_unit_id,
                "candidate_registry_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_hybrid_split(
    split: str,
    *,
    stage: str = "b1-clean",
    resume: bool = False,
    retry_timeouts: bool = False,
    allow_holdout: bool = False,
    preflight_only: bool = False,
) -> dict[str, object]:
    is_holdout = split == "holdout"
    if split not in {"dev", "holdout"}:
        raise ValueError("Phase 11B hybrid execution supports DEV and protected HOLDOUT only")
    if is_holdout and (not allow_holdout or stage != "b2"):
        raise ValueError("protected HOLDOUT execution requires --allow-holdout and Stage B2")
    if split == "dev" and allow_holdout:
        raise ValueError("HOLDOUT authorization cannot be used for DEV")
    if stage not in {"b1-clean", "b2"}:
        raise ValueError("unknown Phase 11B stage")
    if preflight_only and stage != "b2":
        raise ValueError("--preflight-only is available only for Stage B2")
    if stage == "b2":
        stage_name = "hybrid-qwen-v1-stage-b2"
        prompt_template_version = HYBRID_STAGE_B2_PROMPT_TEMPLATE_VERSION
        if is_holdout:
            checkpoint_root = HYBRID_STAGE_B2_HOLDOUT_CHECKPOINT_ROOT
            result_root = HYBRID_STAGE_B2_HOLDOUT_RESULT_ROOT
            config_path = HYBRID_STAGE_B2_HOLDOUT_CONFIG_PATH
            private_metadata_path = HYBRID_STAGE_B2_HOLDOUT_PRIVATE_METADATA_PATH
        else:
            checkpoint_root = HYBRID_STAGE_B2_CLEAN_CHECKPOINT_ROOT
            result_root = HYBRID_STAGE_B2_CLEAN_RESULT_ROOT
            config_path = HYBRID_STAGE_B2_CLEAN_CONFIG_PATH
            private_metadata_path = HYBRID_STAGE_B2_CLEAN_PRIVATE_METADATA_PATH
    else:
        stage_name = "hybrid-qwen-v1"
        prompt_template_version = HYBRID_PROMPT_TEMPLATE_VERSION
        checkpoint_root = HYBRID_STAGE_B1_CLEAN_CHECKPOINT_ROOT
        result_root = HYBRID_STAGE_B1_CLEAN_RESULT_ROOT
        config_path = HYBRID_STAGE_B1_CLEAN_CONFIG_PATH
        private_metadata_path = HYBRID_STAGE_B1_CLEAN_PRIVATE_METADATA_PATH
    prompt_hash = hybrid_prompt_hash(prompt_template_version)
    selection_payload = json.loads(SELECTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if selection_payload.get("selection_fingerprint") != PHASE11_SELECTION_FINGERPRINT:
        raise ValueError("selection fingerprint does not match the locked Phase 11 input")

    if is_holdout:
        frozen = _load_locked_release(
            HOLDOUT_RELEASE_MANIFEST_PATH,
            artifact_type="phase11_holdout_annotation_release",
        )
        if frozen.get("status") != "HOLDOUT_REFERENCE_FROZEN" or frozen.get("record_count") != 40:
            raise ValueError("HOLDOUT reference release is not frozen with exactly 40 records")
        if frozen.get("selection_fingerprint") != PHASE11_SELECTION_FINGERPRINT:
            raise ValueError("HOLDOUT release selection fingerprint is not locked")
        if frozen.get("candidate_policy_version") != CANDIDATE_REGISTRY_VERSION:
            raise ValueError("HOLDOUT release candidate policy is not candidates-v3")
        if frozen.get("candidate_policy_hash") != PHASE11_CANDIDATE_POLICY_HASH:
            raise ValueError("HOLDOUT release candidate policy hash is not locked")
        if hashlib.sha256(HOLDOUT_RELEASE_PATH.read_bytes()).hexdigest() != frozen.get(
            "private_release_sha256"
        ):
            raise ValueError("HOLDOUT private annotation release hash does not match manifest")
        records = _load_holdout_source_records_for_inference()
        if len(records) != 40:
            raise ValueError("Phase 11B HOLDOUT requires exactly 40 records")
        semantic_release_fingerprint = str(frozen["annotation_release_fingerprint"])
        candidate_compatible_fingerprint = _candidate_collection_fingerprint(records)
        if frozen.get("candidate_collection_fingerprint") != candidate_compatible_fingerprint:
            raise ValueError("HOLDOUT candidate-compatible fingerprint does not match source batch")
        selection_fingerprint = PHASE11_SELECTION_FINGERPRINT
    else:
        records = _load_records("dev")
        if len(records) != 80:
            raise ValueError(f"Phase 11B requires exactly 80 DEV records, found {len(records)}")
        semantic_release = _load_locked_release(
            SEMANTIC_RELEASE_PATH,
            artifact_type="phase11_dev_annotation_release",
        )
        candidate_release = _load_locked_release(
            CANDIDATE_COMPATIBLE_RELEASE_PATH,
            artifact_type="phase11_dev_candidate_compatible_release",
        )
        if (
            semantic_release.get("annotation_release_fingerprint")
            != PHASE11_SEMANTIC_RELEASE_FINGERPRINT
        ):
            raise ValueError(
                "semantic annotation release fingerprint is not the locked DEV release"
            )
        if (
            hashlib.sha256(SEMANTIC_RELEASE_PATH.read_bytes()).hexdigest()
            != PHASE11_SEMANTIC_RELEASE_SHA256
        ):
            raise ValueError("semantic annotation release SHA-256 is not the locked DEV release")
        if (
            candidate_release.get("candidate_compatible_release_fingerprint")
            != PHASE11_CANDIDATE_COMPATIBLE_FINGERPRINT
        ):
            raise ValueError("candidate-compatible DEV release fingerprint is not locked")
        if candidate_release.get("candidate_policy_version") != CANDIDATE_REGISTRY_VERSION:
            raise ValueError("candidate policy version is not phase11-candidates-v3")
        if candidate_release.get("candidate_policy_hash") != PHASE11_CANDIDATE_POLICY_HASH:
            raise ValueError("candidate policy hash is not the locked Phase 11B policy")
        if (
            semantic_release.get("record_count") != 80
            or candidate_release.get("record_count") != 80
        ):
            raise ValueError("locked DEV releases do not contain exactly 80 records")
        dev_ids = {record.canonical_unit_id for record in records}
        if any(record.split != "dev" for record in records):
            raise ValueError("hybrid input contained a non-DEV record")
        annotation_errors = [
            error for record in records for error in validate_annotation_record(record, dev_ids)
        ]
        if annotation_errors:
            raise ValueError("locked DEV annotations are invalid: " + "; ".join(annotation_errors))
        provenance_counts = {
            "dual_ai_agreed": sum(
                record.annotation_provenance == "dual_ai_agreed" for record in records
            ),
            "ai_adjudicated_after_independent_second_review": sum(
                record.annotation_provenance == "ai_adjudicated_after_independent_second_review"
                for record in records
            ),
            "human_verified": sum(record.human_verified for record in records),
        }
        if provenance_counts != {
            "dual_ai_agreed": 3,
            "ai_adjudicated_after_independent_second_review": 77,
            "human_verified": 0,
        }:
            raise ValueError("DEV semantic annotation provenance does not match the locked release")
        _validate_candidate_release_records(records, candidate_release)
        semantic_release_fingerprint = PHASE11_SEMANTIC_RELEASE_FINGERPRINT
        candidate_compatible_fingerprint = PHASE11_CANDIDATE_COMPATIBLE_FINGERPRINT
        selection_fingerprint = PHASE11_SELECTION_FINGERPRINT

    if is_holdout:
        locked_config_sha = hashlib.sha256(B2_SELECTED_CONFIG_PATH.read_bytes()).hexdigest()
        selected_config = json.loads(B2_SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
        if locked_config_sha != "0d3736250a3b72f582d75720ea02fbb60a4762f3157006f741520b2f2eaab25e":
            raise ValueError("selected Stage B2 configuration hash is not locked")
        if (
            selected_config.get("stage") != stage_name
            or selected_config.get("template_sha256") != prompt_hash
        ):
            raise ValueError(
                "selected Stage B2 prompt/config does not match the locked configuration"
            )
        if selected_config.get("schema_sha256") != hybrid_schema_hash():
            raise ValueError("selected Stage B2 schema hash does not match runtime schema")
        if selected_config.get("candidate_policy_hash") != PHASE11_CANDIDATE_POLICY_HASH:
            raise ValueError("selected Stage B2 candidate policy hash does not match candidates-v3")

    if preflight_only:
        checkpoint_status = ExtractionCheckpointStore(checkpoint_root).status()
        result_files = tuple(
            path for path in result_root.glob("*.json") if path.name != "experiment.json"
        )
        if (
            checkpoint_status
            != {
                "completed": 0,
                "incomplete": 0,
                "failed": 0,
                "corrupt": 0,
            }
            or result_files
        ):
            raise ValueError("Stage B2 clean namespace is not empty")
        from kawaneen.extraction.tokenizer import pinned_local_tokenizer

        pinned_local_tokenizer().preflight()
        config = _hybrid_configuration_metadata(
            state="ready_not_executed",
            run=None,
            stage=stage_name,
            prompt_template_version=prompt_template_version,
            prompt_hash=prompt_hash,
            checkpoint_root=checkpoint_root,
            result_root=result_root,
            split=split,
            record_count=len(records),
            semantic_release_fingerprint=semantic_release_fingerprint,
            candidate_compatible_release_fingerprint=candidate_compatible_fingerprint,
            holdout_access=int(is_holdout),
            reference_labels_loaded=0,
            evaluation_performed=0,
        )
        write_text_free_json(config_path, config)
        write_private_json(private_metadata_path, config)
        return {
            "split": split,
            "stage": stage_name,
            "record_count": len(records),
            "completed": 0,
            "pending": len(records),
            "provider_calls_attempted": 0,
            "holdout_access": int(is_holdout),
        }

    # Both preflights are local-only: tokenizer cache loading and Ollama /api/tags.
    from kawaneen.extraction.tokenizer import pinned_local_tokenizer

    pinned_local_tokenizer().preflight()
    provider = OllamaExtractionProvider(prompt_template_version=prompt_template_version)
    provider.preflight()
    config = _hybrid_configuration_metadata(
        state="ready_not_executed",
        run=None,
        stage=stage_name,
        prompt_template_version=prompt_template_version,
        prompt_hash=prompt_hash,
        checkpoint_root=checkpoint_root,
        result_root=result_root,
        split=split,
        record_count=len(records),
        semantic_release_fingerprint=semantic_release_fingerprint,
        candidate_compatible_release_fingerprint=candidate_compatible_fingerprint,
        holdout_access=int(is_holdout),
        reference_labels_loaded=0,
        evaluation_performed=0,
    )
    write_text_free_json(config_path, config)
    write_private_json(private_metadata_path, config)
    result = run_hybrid_records(
        records,
        provider,
        checkpoint_root=checkpoint_root,
        result_root=result_root,
        selection_fingerprint=selection_fingerprint,
        semantic_release_fingerprint=semantic_release_fingerprint,
        candidate_compatible_release_fingerprint=candidate_compatible_fingerprint,
        prompt_hash=prompt_hash,
        schema_hash=hybrid_schema_hash(),
        qwen_model=HYBRID_QWEN_MODEL,
        qwen_digest=HYBRID_QWEN_OLLAMA_DIGEST,
        tokenizer_revision=HYBRID_QWEN_TOKENIZER_REVISION,
        resume=resume,
        retry_timeouts=retry_timeouts,
        accept_field_local_diagnostics=True,
        experiment_stage=stage_name,
    )
    completed_config = _hybrid_configuration_metadata(
        state="complete",
        run=result,
        stage=stage_name,
        prompt_template_version=prompt_template_version,
        prompt_hash=prompt_hash,
        checkpoint_root=checkpoint_root,
        result_root=result_root,
        split=split,
        record_count=len(records),
        semantic_release_fingerprint=semantic_release_fingerprint,
        candidate_compatible_release_fingerprint=candidate_compatible_fingerprint,
        holdout_access=int(is_holdout),
        reference_labels_loaded=0,
        evaluation_performed=0,
    )
    write_text_free_json(config_path, completed_config)
    write_private_json(private_metadata_path, completed_config)
    return {"split": split, "extractor": stage_name, **result}


def _load_locked_release(path: Path, *, artifact_type: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"locked Phase 11B release is unavailable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected Phase 11B release artifact: {path}")
    typed_payload = cast(dict[str, object], payload)
    if typed_payload.get("artifact_type") != artifact_type:
        raise ValueError(f"unexpected Phase 11B release artifact: {path}")
    return typed_payload


def _validate_candidate_release_records(
    records: list[AnnotationRecord], release: dict[str, object]
) -> None:
    raw_fingerprints = release.get("record_fingerprints")
    if not isinstance(raw_fingerprints, list):
        raise ValueError("candidate-compatible release has no record fingerprints")
    expected: dict[str, dict[str, object]] = {}
    for raw_item in cast(list[object], raw_fingerprints):
        if not isinstance(raw_item, dict):
            continue
        item = cast(dict[str, object], raw_item)
        record_id = item.get("canonical_unit_id")
        if isinstance(record_id, str):
            expected[record_id] = item
    if set(expected) != {record.canonical_unit_id for record in records}:
        raise ValueError("candidate-compatible release IDs do not match DEV")
    for record in records:
        item = expected[record.canonical_unit_id]
        encoded = json.dumps(
            record.candidate_registry.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != item.get(
            "candidate_registry_sha256"
        ):
            raise ValueError(f"candidate registry fingerprint mismatch: {record.canonical_unit_id}")


def _hybrid_configuration_metadata(
    *,
    state: str,
    run: dict[str, object] | None,
    stage: str = "hybrid-qwen-v1",
    prompt_template_version: str = HYBRID_PROMPT_TEMPLATE_VERSION,
    prompt_hash: str | None = None,
    checkpoint_root: Path = HYBRID_CHECKPOINT_ROOT,
    result_root: Path = HYBRID_RESULT_ROOT,
    split: str = "dev",
    record_count: int = 80,
    semantic_release_fingerprint: str = PHASE11_SEMANTIC_RELEASE_FINGERPRINT,
    candidate_compatible_release_fingerprint: str = PHASE11_CANDIDATE_COMPATIBLE_FINGERPRINT,
    holdout_access: int = 0,
    reference_labels_loaded: int = 0,
    evaluation_performed: int = 0,
) -> dict[str, object]:
    selected_prompt_hash = prompt_hash or hybrid_prompt_hash(prompt_template_version)
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "phase11_hybrid_dev_configuration",
        "stage": stage,
        "stage_experiment": "Stage B2 is the final DEV prompt experiment."
        if stage == "hybrid-qwen-v1-stage-b2"
        else None,
        "lifecycle_state": state,
        "split": split,
        "record_count": record_count,
        "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
        "semantic_release_fingerprint": semantic_release_fingerprint,
        "semantic_release_sha256": (PHASE11_SEMANTIC_RELEASE_SHA256 if split == "dev" else None),
        "candidate_policy_version": CANDIDATE_REGISTRY_VERSION,
        "candidate_policy_hash": PHASE11_CANDIDATE_POLICY_HASH,
        "candidate_compatible_release_fingerprint": candidate_compatible_release_fingerprint,
        "template_version": prompt_template_version,
        "template_sha256": selected_prompt_hash,
        "schema_sha256": hybrid_schema_hash(),
        "hf_model": HYBRID_QWEN_HF_ID,
        "hf_revision": HYBRID_QWEN_HF_REVISION,
        "model": HYBRID_QWEN_MODEL,
        "ollama_tag": HYBRID_QWEN_MODEL,
        "ollama_digest": HYBRID_QWEN_OLLAMA_DIGEST,
        "tokenizer_revision": HYBRID_QWEN_TOKENIZER_REVISION,
        "runtime_settings": HYBRID_RUNTIME_SETTINGS,
        "checkpoint_root": checkpoint_root.as_posix(),
        "result_root": result_root.as_posix(),
        "automatic_retries": 0,
        "holdout_access": holdout_access,
        "reference_labels_loaded": reference_labels_loaded,
        "evaluation_performed": evaluation_performed,
        "model_calls": 0,
    }
    if run is not None:
        payload["run"] = run
        payload["model_calls"] = run.get("model_calls", 0)
    return payload


def evaluate_split(extractor: str, split: str, *, allow_holdout: bool = False) -> dict[str, object]:
    _guard_split(split, allow_holdout)
    if extractor not in {"deterministic-v1", "hybrid-qwen-v1"}:
        raise ValueError("unknown extraction configuration")
    records = _load_records("dev" if split == "smoke" else cast(Literal["dev", "holdout"], split))
    selected = [
        record
        for record in records
        if record.split == ("dev" if split == "smoke" else split)
        and (split != "smoke" or record.smoke)
    ]
    if not selected:
        raise ValueError("no records selected")
    if not any(is_human_gold(record) for record in selected):
        return {
            "split": split,
            "extractor": extractor,
            "semantic_performance_available": False,
            "reason": "annotation records are not human-verified",
            "record_count": len(selected),
        }
    raise RuntimeError(
        "human-gold evaluation orchestration is intentionally deferred until annotation review"
    )


def next_dev_annotation() -> dict[str, object]:
    """Return the next unreviewed DEV record, including private source text."""

    record, _, _ = next_dev_annotation_context()
    return cast(dict[str, object], record.model_dump(mode="json"))


def next_dev_annotation_context() -> tuple[AnnotationRecord, int, int]:
    """Return the next DEV record and its one-based position in the sorted DEV pack."""

    records = sorted(_load_records("dev"), key=lambda record: record.canonical_unit_id)
    for index, record in enumerate(records):
        if record.annotation_status in {"unreviewed", "in_review"}:
            return record, index + 1, len(records)
    raise ValueError("all DEV records are reviewed")


def annotation_progress(split: str) -> dict[str, int | str]:
    """Report annotation progress without exposing private source text."""

    if split != "dev":
        raise ValueError("annotation progress is DEV-only")
    records = _load_records("dev")
    selection_ids = {record.canonical_unit_id for record in records}
    invalid = sum(bool(validate_annotation_record(record, selection_ids)) for record in records)
    reviewed = sum(record.annotation_status == "reviewed" for record in records)
    pending = sum(record.annotation_status in {"unreviewed", "in_review"} for record in records)
    human_verified = sum(record.human_verified for record in records)
    return {
        "split": "dev",
        "total": len(records),
        "reviewed": reviewed,
        "human_verified": human_verified,
        "remaining": pending,
        "invalid": invalid,
    }


def save_dev_annotation(
    record_id: str,
    annotation_path: Path,
    *,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
) -> dict[str, object]:
    """Validate and atomically save one DEV annotation update."""

    update = AnnotationUpdate.model_validate(
        json.loads(annotation_path.read_text(encoding="utf-8"))
    )
    return save_dev_annotation_update(
        record_id,
        update,
        annotation_root=annotation_root,
        selection_manifest_path=selection_manifest_path,
    )


def _prepare_dev_annotation_update(
    record_id: str,
    update: AnnotationUpdate,
    *,
    annotation_root: Path,
    selection_manifest_path: Path,
) -> tuple[AnnotationRecord, AnnotationRecord, list[str]]:
    records = _load_records(
        "dev",
        annotation_root=annotation_root,
        selection_manifest_path=selection_manifest_path,
    )
    record = next((item for item in records if item.canonical_unit_id == record_id), None)
    if record is None:
        raise ValueError("record is not in the DEV annotation pack")
    updated = record.model_copy(
        update={
            "human_annotations": update.human_annotations,
            "annotation_status": update.annotation_status,
            "annotation_provenance": (
                "human_adjudicated" if update.human_verified else record.annotation_provenance
            ),
            "human_verified": update.human_verified,
        }
    )
    errors = validate_annotation_record(updated, {item.canonical_unit_id for item in records})
    return record, updated, errors


def validate_dev_annotation_update(
    record_id: str,
    update: AnnotationUpdate,
    *,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
) -> list[str]:
    """Validate one private DEV update without persisting it."""

    _, _, errors = _prepare_dev_annotation_update(
        record_id,
        update,
        annotation_root=annotation_root,
        selection_manifest_path=selection_manifest_path,
    )
    return errors


def save_dev_annotation_update(
    record_id: str,
    update: AnnotationUpdate,
    *,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
) -> dict[str, object]:
    """Validate and atomically save one in-memory DEV annotation update."""

    _, updated, errors = _prepare_dev_annotation_update(
        record_id,
        update,
        annotation_root=annotation_root,
        selection_manifest_path=selection_manifest_path,
    )
    if errors:
        raise ValueError("annotation rejected: " + "; ".join(errors))
    write_private_json(
        _private_record_path(record_id, annotation_root),
        cast(dict[str, object], updated.model_dump(mode="json")),
    )
    return {
        "record_id": record_id,
        "split": "dev",
        "annotation_status": updated.annotation_status,
        "human_verified": updated.human_verified,
    }


def _annotation_contract() -> dict[str, object]:
    return {
        "schema_version": "phase11-proposal-v1",
        "allowed_modalities": ["obligation", "prohibition", "permission"],
        "semantic_span_representation": {
            "fields": ["text", "occurrence"],
            "offsets": "derived and validated from canonical text codepoints",
            "exact_substring_required": True,
        },
        "candidate_reference_rules": {
            "allowed_id_patterns": ["T###", "M###", "P###", "A###", "R###"],
            "must_exist_in_candidate_registry": True,
            "dates_money_percentages_are_candidates_not_automatic_classifications": True,
        },
        "target_fields": [
            "regulated_entities",
            "rules",
            "actor",
            "action",
            "conditions",
            "exceptions",
            "penalties",
            "deadline_refs",
            "effective_date_refs",
            "monetary_threshold_refs",
            "percentage_threshold_refs",
        ],
        "guidelines_version": "phase11-annotation-contract-v1",
        "guidelines": [
            "semantic spans must be exact substrings",
            "use {text, occurrence} rather than manually entered offsets",
            "do not paraphrase",
            "do not infer missing facts",
            "issuing_authority is metadata-only",
            "zero-rule clauses are valid",
            "one clause may contain multiple rules",
            "deterministic dates/money/percentages are candidates, not automatic classifications",
        ],
    }


def export_dev_annotation_batch(
    *,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
    output_path: Path = DEV_BATCH_PATH,
) -> dict[str, object]:
    """Export the complete private DEV annotation pack for independent review."""

    records = _load_records(
        "dev",
        annotation_root=annotation_root,
        selection_manifest_path=selection_manifest_path,
    )
    if len(records) != 80:
        raise ValueError(f"expected exactly 80 DEV records, found {len(records)}")
    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    record_payloads: list[dict[str, object]] = []
    for record in records:
        record_payload = cast(dict[str, object], record.model_dump(mode="json"))
        record_payload["human_verified"] = False
        record_payloads.append(record_payload)
    payload = {
        "schema_version": "phase11-dev-annotation-batch-v1",
        "artifact_type": "phase11_dev_annotation_batch",
        "split": "dev",
        "selection_fingerprint": str(selection_payload["selection_fingerprint"]),
        "record_count": len(records),
        "review_purpose": "independent_ai_review",
        "human_verified": False,
        "annotation_contract": _annotation_contract(),
        "records": record_payloads,
    }
    write_private_json(output_path, payload)
    return {
        "path": output_path.as_posix(),
        "record_count": len(records),
        "dev_records": len(records),
        "holdout_records": 0,
        "selection_fingerprint": payload["selection_fingerprint"],
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def export_dev_annotation_batch_v2(
    *,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
    output_path: Path = DEV_BATCH_V2_PATH,
) -> dict[str, object]:
    """Export the fresh v2 DEV selection for independent review."""

    records = _load_records(
        "dev", annotation_root=annotation_root, selection_manifest_path=selection_manifest_path
    )
    if len(records) != 80:
        raise ValueError(f"expected exactly 80 DEV records, found {len(records)}")
    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if selection_payload.get("selection_version") != PHASE11_SELECTION_VERSION:
        raise ValueError("active selection is not Phase-11 v2")
    record_payloads: list[dict[str, object]] = []
    for record in records:
        if (
            record.annotation_status != "unreviewed"
            or record.annotation_provenance != "unreviewed"
            or record.human_annotations is not None
            or record.human_verified
        ):
            raise ValueError("v2 batch export requires all records to begin unreviewed")
        record_payloads.append(cast(dict[str, object], record.model_dump(mode="json")))
    payload = {
        "schema_version": "phase11-dev-annotation-batch-v2",
        "artifact_type": "phase11_dev_annotation_batch",
        "split": "dev",
        "selection_version": PHASE11_SELECTION_VERSION,
        "eligibility_policy_version": PHASE11_ELIGIBILITY_POLICY_VERSION,
        "candidate_registry_version": CANDIDATE_REGISTRY_VERSION,
        "selection_fingerprint": str(selection_payload["selection_fingerprint"]),
        "record_count": len(record_payloads),
        "review_purpose": "independent_ai_review",
        "annotation_provenance": "unreviewed",
        "human_verified": False,
        "annotation_contract": _annotation_contract(),
        "records": record_payloads,
    }
    write_private_json(output_path, payload)
    return {
        "path": output_path.as_posix(),
        "record_count": len(record_payloads),
        "dev_records": len(record_payloads),
        "holdout_records": 0,
        "selection_fingerprint": payload["selection_fingerprint"],
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def export_holdout_annotation_batch(
    *,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
    output_path: Path = HOLDOUT_BATCH_PATH,
) -> dict[str, object]:
    """Export protected HOLDOUT source material for annotation, without predictions."""

    records = _load_records(
        "holdout",
        annotation_root=annotation_root,
        selection_manifest_path=selection_manifest_path,
    )
    if len(records) != 40:
        raise ValueError(f"expected exactly 40 HOLDOUT records, found {len(records)}")
    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if selection_payload.get("selection_version") != PHASE11_SELECTION_VERSION:
        raise ValueError("HOLDOUT export requires the active Phase-11 v2 selection")
    record_payloads: list[dict[str, object]] = []
    candidate_counts: dict[str, int] = {}
    candidate_records: dict[str, int] = {}
    for record in records:
        if (
            record.annotation_status != "unreviewed"
            or record.annotation_provenance != "unreviewed"
            or record.human_annotations is not None
            or record.human_verified
        ):
            raise ValueError("HOLDOUT export requires all records to begin unreviewed")
        # Rebuild only the exported registry so the batch is explicitly candidates-v3;
        # the private annotation record itself remains untouched.
        registry = build_candidate_registry(
            record.canonical_text,
            canonical_unit_id=record.canonical_unit_id,
            document_id=record.document_id,
        )
        payload = cast(dict[str, object], record.model_dump(mode="json"))
        payload["candidate_registry"] = registry.model_dump(mode="json")
        payload["annotation_status"] = "unreviewed"
        payload["annotation_provenance"] = "unreviewed"
        payload["human_annotations"] = None
        payload["human_verified"] = False
        record_payloads.append(payload)
        seen: set[str] = set()
        for candidate in registry.candidates:
            kind = candidate.candidate_type.value
            candidate_counts[kind] = candidate_counts.get(kind, 0) + 1
            seen.add(kind)
        for kind in seen:
            candidate_records[kind] = candidate_records.get(kind, 0) + 1
    payload = {
        "schema_version": "phase11-holdout-annotation-batch-v1",
        "artifact_type": "phase11_holdout_annotation_batch",
        "split": "holdout",
        "selection_version": PHASE11_SELECTION_VERSION,
        "eligibility_policy_version": PHASE11_ELIGIBILITY_POLICY_VERSION,
        "candidate_registry_version": CANDIDATE_REGISTRY_VERSION,
        "selection_fingerprint": str(selection_payload["selection_fingerprint"]),
        "record_count": len(record_payloads),
        "review_purpose": "protected_source_only_annotation",
        "annotation_provenance": "unreviewed",
        "human_verified": False,
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "records_containing_candidate_type": dict(sorted(candidate_records.items())),
        "annotation_contract": _annotation_contract(),
        "records": record_payloads,
    }
    write_private_json(output_path, payload)
    return {
        "path": output_path.as_posix(),
        "record_count": len(record_payloads),
        "dev_records": 0,
        "holdout_records": len(record_payloads),
        "selection_fingerprint": payload["selection_fingerprint"],
        "candidate_registry_version": CANDIDATE_REGISTRY_VERSION,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def freeze_stage_b2_configuration(
    *,
    comparison_path: Path = B2_SELECTION_ARTIFACT_PATH,
    b2_config_path: Path = B2_SELECTED_CONFIG_PATH,
    b2_evaluation_path: Path = B2_EVALUATION_PATH,
    semantic_release_path: Path = SEMANTIC_RELEASE_PATH,
    output_path: Path = PHASE11_SELECTED_CONFIGURATION_PATH,
) -> dict[str, object]:
    """Freeze the selected B2 DEV configuration as text-free tracked metadata."""

    comparison = cast(dict[str, object], json.loads(comparison_path.read_text(encoding="utf-8")))
    hashes = cast(dict[str, object], comparison.get("hashes", {}))
    config = cast(dict[str, object], json.loads(b2_config_path.read_text(encoding="utf-8")))
    if comparison.get("selection") != "SELECT_STAGE_B2_EXPERIMENTAL":
        raise ValueError("B2 selection artifact does not select Stage B2")
    if comparison.get("holdout_recommendation") != "FREEZE_B2_AND_RUN_HOLDOUT_ONCE":
        raise ValueError("B2 selection artifact does not authorize one protected HOLDOUT run")
    if comparison.get("record_count") != 80 or comparison.get("reference_status") == "human_gold":
        raise ValueError("B2 selection artifact has invalid DEV scope or reference status")
    if config.get("stage") != "hybrid-qwen-v1-stage-b2" or config.get("split") != "dev":
        raise ValueError("B2 configuration is not the selected DEV stage")
    if config.get("record_count") != 80 or config.get("lifecycle_state") != "complete":
        raise ValueError("B2 configuration is not a complete 80-record DEV experiment")
    if config.get("holdout_access") != 0:
        raise ValueError("B2 configuration records HOLDOUT access")
    config_sha = hashlib.sha256(b2_config_path.read_bytes()).hexdigest()
    evaluation_sha = hashlib.sha256(b2_evaluation_path.read_bytes()).hexdigest()
    release_sha = hashlib.sha256(semantic_release_path.read_bytes()).hexdigest()
    if hashes.get("b2_config_sha256") != config_sha:
        raise ValueError("B2 configuration SHA does not match the selection artifact")
    if hashes.get("b2_evaluation_sha256") != evaluation_sha:
        raise ValueError("B2 evaluation SHA does not match the selection artifact")
    if release_sha != PHASE11_SEMANTIC_RELEASE_SHA256:
        raise ValueError("semantic annotation release SHA does not match the locked DEV release")
    if config.get("candidate_policy_version") != CANDIDATE_REGISTRY_VERSION:
        raise ValueError("B2 configuration candidate policy is not candidates-v3")
    if config.get("candidate_policy_hash") != PHASE11_CANDIDATE_POLICY_HASH:
        raise ValueError("B2 configuration candidate policy hash is not locked")
    runtime = cast(dict[str, object], config.get("runtime_settings", {}))
    manifest = {
        "schema_version": 1,
        "artifact_type": "phase11_selected_configuration",
        "selection": "SELECT_STAGE_B2_EXPERIMENTAL",
        "status": "DEV_FROZEN_PENDING_PROTECTED_HOLDOUT",
        "split": "dev",
        "records": 80,
        "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
        "template_version": config.get("template_version"),
        "template_sha256": config.get("template_sha256"),
        "schema_sha256": config.get("schema_sha256"),
        "candidate_policy_version": CANDIDATE_REGISTRY_VERSION,
        "candidate_policy_hash": PHASE11_CANDIDATE_POLICY_HASH,
        "candidate_compatible_release_fingerprint": PHASE11_CANDIDATE_COMPATIBLE_FINGERPRINT,
        "hf_model": config.get("hf_model"),
        "hf_revision": config.get("hf_revision"),
        "model": config.get("model"),
        "ollama_tag": config.get("ollama_tag"),
        "ollama_digest": config.get("ollama_digest"),
        "runtime_settings": runtime,
        "b2_result_set_sha256": hashes.get("b2_result_set_sha256"),
        "b2_evaluation_sha256": evaluation_sha,
        "b2_configuration_sha256": config_sha,
        "b1_vs_b2_selection_artifact_sha256": hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest(),
        "semantic_annotation_release_sha256": release_sha,
        "semantic_annotation_release_fingerprint": PHASE11_SEMANTIC_RELEASE_FINGERPRINT,
        "selection_rationale": comparison.get("selection_reason"),
        "experimental_disclaimer": "AI-reviewed/adjudicated DEV reference; not human gold.",
        "full_rule_exact_f1_dev": 0.0,
        "hard_safety_gates": {
            "unsupported_span_acceptance_rate": 0.0,
            "invalid_candidate_reference_acceptance_rate": 0.0,
            "final_schema_validity_rate_completed": 1.0,
            "provenance_completeness_rate_completed": 1.0,
        },
        "human_gold": False,
        "holdout_model_inference": 0,
        "holdout_evaluation": 0,
    }
    write_text_free_json(output_path, cast(dict[str, object], manifest))
    return {
        "path": output_path.as_posix(),
        "status": manifest["status"],
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "records": 80,
        "holdout_model_inference": 0,
        "holdout_evaluation": 0,
    }


def write_dev_candidate_audit_v2(
    *,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
    output_path: Path = CANDIDATE_AUDIT_V2_PATH,
) -> dict[str, object]:
    """Write a private, bounded deterministic candidate sample with source context."""

    records = _load_records(
        "dev", annotation_root=annotation_root, selection_manifest_path=selection_manifest_path
    )
    if len(records) != 80:
        raise ValueError("candidate audit requires exactly 80 DEV records")
    by_type: dict[str, list[dict[str, object]]] = {
        "temporal": [],
        "regulation": [],
        "monetary": [],
        "percentage": [],
    }
    for record in records:
        for candidate in record.candidate_registry.candidates:
            kind = candidate.candidate_type.value
            if kind not in by_type or len(by_type[kind]) >= 20:
                continue
            start = max(0, candidate.span.start_char - 80)
            end = min(len(record.canonical_text), candidate.span.end_char + 80)
            by_type[kind].append(
                {
                    "canonical_unit_id": record.canonical_unit_id,
                    "document_id": record.document_id,
                    "candidate_id": candidate.candidate_id,
                    "candidate_type": kind,
                    "raw_exact_text": candidate.raw_exact_text,
                    "start_char": candidate.span.start_char,
                    "end_char": candidate.span.end_char,
                    "normalized": candidate.normalized.model_dump(mode="json"),
                    "normalization_status": candidate.normalization_status.value,
                    "source_context": record.canonical_text[start:end],
                }
            )
    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "phase11-dev-candidate-audit-v2",
        "artifact_type": "phase11_deterministic_candidate_audit",
        "selection_fingerprint": selection_payload["selection_fingerprint"],
        "split": "dev",
        "model_calls": 0,
        "candidates": by_type,
    }
    write_private_json(output_path, payload)
    return {
        "path": output_path.as_posix(),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "candidate_counts": {key: len(value) for key, value in by_type.items()},
    }


def import_reviewed_dev(
    reviewed_path: Path,
    *,
    partial: bool = False,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
) -> dict[str, object]:
    """Import an explicitly supplied independent-AI DEV review without human gold."""

    payload_value = json.loads(reviewed_path.read_text(encoding="utf-8"))
    if not isinstance(payload_value, dict):
        raise ValueError("reviewed batch must be a JSON object")
    payload = cast(dict[str, object], payload_value)
    batch_schema = payload.get("schema_version")
    if batch_schema not in {
        "phase11-dev-annotation-batch-v1",
        "phase11-dev-annotation-batch-v2",
    }:
        raise ValueError("reviewed batch schema_version is invalid")
    if payload.get("annotation_provenance") != "independent_ai_review":
        raise ValueError("reviewed batch annotation_provenance must be independent_ai_review")
    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if batch_schema == "phase11-dev-annotation-batch-v2":
        if selection_payload.get("selection_version") != PHASE11_SELECTION_VERSION:
            raise ValueError("v2 reviewed batch requires the active Phase-11 v2 selection")
        if payload.get("selection_version") != PHASE11_SELECTION_VERSION:
            raise ValueError("v2 reviewed batch selection version is invalid")
        if payload.get("candidate_registry_version") != CANDIDATE_REGISTRY_VERSION:
            raise ValueError("v2 reviewed batch candidate registry version is invalid")
    expected_fingerprint = str(selection_payload["selection_fingerprint"])
    if payload.get("selection_fingerprint") != expected_fingerprint:
        raise ValueError("reviewed batch selection fingerprint does not match DEV selection")
    raw_records_value = payload.get("records")
    if not isinstance(raw_records_value, list):
        raise ValueError("reviewed batch records must be a list")
    raw_items = cast(list[object], raw_records_value)
    if any(not isinstance(item, dict) for item in raw_items):
        raise ValueError("reviewed batch contains a malformed record")
    raw_records = [cast(dict[str, object], item) for item in raw_items]
    raw_ids = [str(item.get("canonical_unit_id")) for item in raw_records]
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("reviewed batch contains duplicate DEV IDs")

    current_records = _load_records(
        "dev",
        annotation_root=annotation_root,
        selection_manifest_path=selection_manifest_path,
    )
    current_by_id = {record.canonical_unit_id: record for record in current_records}
    incoming_ids = set(raw_ids)
    unknown_ids = incoming_ids - set(current_by_id)
    if unknown_ids:
        raise ValueError(f"reviewed batch contains IDs not in DEV: {sorted(unknown_ids)}")
    missing_ids = set(current_by_id) - incoming_ids
    if missing_ids and not partial:
        raise ValueError(f"reviewed batch is missing DEV IDs: {len(missing_ids)}")
    if not partial and len(raw_records) != len(current_by_id):
        raise ValueError("reviewed batch does not contain exactly the expected DEV records")
    if not raw_records:
        raise ValueError("reviewed batch contains no records")

    updates: list[tuple[Path, dict[str, object]]] = []
    expected_ids = set(current_by_id)
    for raw in raw_records:
        if raw.get("annotation_provenance") != "independent_ai_review":
            raise ValueError("each imported record must declare independent_ai_review provenance")
        if raw.get("human_verified") is True:
            raise ValueError("independent-AI imports cannot set human_verified=true")
        incoming = AnnotationRecord.model_validate(
            {
                **raw,
                "annotation_status": "independent_ai_review",
                "annotation_provenance": "independent_ai_review",
                "human_verified": False,
            }
        )
        current = current_by_id[incoming.canonical_unit_id]
        for field_name in (
            "document_id",
            "canonical_text",
            "source_provenance",
            "source_fingerprint",
            "strata",
            "smoke",
            "candidate_registry",
        ):
            if getattr(incoming, field_name) != getattr(current, field_name):
                raise ValueError(
                    f"record {incoming.canonical_unit_id} changed immutable field {field_name}"
                )
        if incoming.human_annotations is None:
            raise ValueError(
                f"record {incoming.canonical_unit_id} has no human_annotations proposal"
            )
        updated = current.model_copy(
            update={
                "human_annotations": incoming.human_annotations,
                "annotation_status": "independent_ai_review",
                "annotation_provenance": "independent_ai_review",
                "human_verified": False,
            }
        )
        errors = validate_annotation_record(updated, expected_ids)
        if errors:
            raise ValueError(
                f"record {incoming.canonical_unit_id} annotation rejected: {'; '.join(errors)}"
            )
        target = _private_record_path(incoming.canonical_unit_id, annotation_root)
        updates.append((target, cast(dict[str, object], updated.model_dump(mode="json"))))

    annotation_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".phase11-import-", dir=annotation_root
    ) as staging_name:
        staging_root = Path(staging_name)
        staged: list[tuple[Path, Path]] = []
        for target, record_payload in updates:
            staged_path = staging_root / target.name
            write_private_json(staged_path, record_payload)
            staged.append((staged_path, target))
        for staged_path, target in staged:
            os.replace(staged_path, target)
    return {
        "split": "dev",
        "imported_records": len(updates),
        "partial": partial,
        "annotation_provenance": "independent_ai_review",
        "human_verified": False,
    }


def import_reviewed_holdout(
    reviewed_path: Path,
    *,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
) -> dict[str, object]:
    """Import one explicit HOLDOUT AI review with no human-gold promotion."""

    payload_value = json.loads(reviewed_path.read_text(encoding="utf-8"))
    if not isinstance(payload_value, dict):
        raise ValueError("reviewed HOLDOUT batch must be a JSON object")
    payload = cast(dict[str, object], payload_value)
    if payload.get("schema_version") != "phase11-holdout-annotation-batch-v1":
        raise ValueError("reviewed HOLDOUT batch schema_version is invalid")
    if payload.get("split") != "holdout":
        raise ValueError("reviewed HOLDOUT batch must declare split=holdout")
    if payload.get("selection_version") != PHASE11_SELECTION_VERSION:
        raise ValueError("reviewed HOLDOUT batch selection version is invalid")
    if payload.get("candidate_registry_version") != CANDIDATE_REGISTRY_VERSION:
        raise ValueError("reviewed HOLDOUT batch candidate policy is not candidates-v3")
    if payload.get("annotation_provenance") != "independent_ai_review":
        raise ValueError("reviewed HOLDOUT provenance must be independent_ai_review")
    if payload.get("human_verified") is True:
        raise ValueError("HOLDOUT AI imports cannot set human_verified=true")
    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if payload.get("selection_fingerprint") != selection_payload.get("selection_fingerprint"):
        raise ValueError("reviewed HOLDOUT selection fingerprint does not match selection")
    raw_records_value = payload.get("records")
    if not isinstance(raw_records_value, list):
        raise ValueError("reviewed HOLDOUT records must be a list")
    raw_items = cast(list[object], raw_records_value)
    if any(not isinstance(item, dict) for item in raw_items):
        raise ValueError("reviewed HOLDOUT contains a malformed record")
    raw_records = [cast(dict[str, object], item) for item in raw_items]
    current_records = _load_records(
        "holdout",
        annotation_root=annotation_root,
        selection_manifest_path=selection_manifest_path,
    )
    if len(current_records) != 40 or len(raw_records) != 40:
        raise ValueError("reviewed HOLDOUT import requires exactly 40 records")
    current_by_id = {record.canonical_unit_id: record for record in current_records}
    raw_ids = [str(item.get("canonical_unit_id")) for item in raw_records]
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("reviewed HOLDOUT contains duplicate IDs")
    incoming_ids = set(raw_ids)
    if incoming_ids != set(current_by_id):
        raise ValueError("reviewed HOLDOUT IDs do not exactly match protected HOLDOUT")

    updates: list[tuple[Path, dict[str, object]]] = []
    expected_ids = set(current_by_id)
    for raw in raw_records:
        if raw.get("annotation_provenance") != "independent_ai_review":
            raise ValueError("each HOLDOUT record must declare independent_ai_review")
        if raw.get("human_verified") is True:
            raise ValueError("HOLDOUT AI records cannot set human_verified=true")
        incoming = AnnotationRecord.model_validate(
            {
                **raw,
                "annotation_status": "independent_ai_review",
                "annotation_provenance": "independent_ai_review",
                "human_verified": False,
            }
        )
        current = current_by_id[incoming.canonical_unit_id]
        for field_name in (
            "document_id",
            "canonical_text",
            "source_provenance",
            "source_fingerprint",
            "strata",
            "smoke",
        ):
            if getattr(incoming, field_name) != getattr(current, field_name):
                raise ValueError(
                    f"record {incoming.canonical_unit_id} changed immutable field {field_name}"
                )
        expected_registry = build_candidate_registry(
            current.canonical_text,
            canonical_unit_id=current.canonical_unit_id,
            document_id=current.document_id,
        )
        if incoming.candidate_registry != expected_registry:
            raise ValueError(
                f"record {incoming.canonical_unit_id} candidate registry is not candidates-v3"
            )
        if incoming.human_annotations is None:
            raise ValueError(
                f"record {incoming.canonical_unit_id} has no human_annotations proposal"
            )
        updated = current.model_copy(
            update={
                "candidate_registry": incoming.candidate_registry,
                "human_annotations": incoming.human_annotations,
                "annotation_status": "independent_ai_review",
                "annotation_provenance": "independent_ai_review",
                "human_verified": False,
            }
        )
        errors = validate_annotation_record(updated, expected_ids)
        if errors:
            raise ValueError(
                f"record {incoming.canonical_unit_id} annotation rejected: {'; '.join(errors)}"
            )
        updates.append(
            (
                _private_record_path(incoming.canonical_unit_id, annotation_root),
                cast(dict[str, object], updated.model_dump(mode="json")),
            )
        )

    annotation_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".phase11-holdout-import-", dir=annotation_root
    ) as name:
        staging_root = Path(name)
        staged: list[tuple[Path, Path]] = []
        for target, record_payload in updates:
            staged_path = staging_root / target.name
            write_private_json(staged_path, record_payload)
            staged.append((staged_path, target))
        for staged_path, target in staged:
            os.replace(staged_path, target)
    return {
        "split": "holdout",
        "imported_records": len(updates),
        "annotation_provenance": "independent_ai_review",
        "human_verified": False,
    }


def _load_review_records(path: Path, *, expected_split: str) -> list[dict[str, object]]:
    payload_value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_value, dict):
        raise ValueError(f"review artifact must be a JSON object: {path}")
    payload = cast(dict[str, object], payload_value)
    if payload.get("split") != expected_split:
        raise ValueError(f"review artifact split is not {expected_split}: {path}")
    raw = payload.get("records")
    if not isinstance(raw, list):
        raise ValueError(f"review artifact records are malformed: {path}")
    raw_items = cast(list[object], raw)
    if any(not isinstance(item, dict) for item in raw_items):
        raise ValueError(f"review artifact records are malformed: {path}")
    records: list[dict[str, object]] = []
    for item in raw_items:
        if isinstance(item, dict):
            records.append(cast(dict[str, object], item))
    return records


def import_adjudicated_holdout(
    adjudication_path: Path,
    *,
    disagreement_path: Path | None = HOLDOUT_DISAGREEMENT_PATH,
    first_review_path: Path = HOLDOUT_FIRST_REVIEW_PATH,
    second_review_path: Path = HOLDOUT_SECOND_REVIEW_PATH,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
    expected_sha256: str | None = PHASE11_HOLDOUT_ADJUDICATION_SHA256,
    verify_review_hashes: bool = True,
) -> dict[str, object]:
    """Apply the sealed 39-record HOLDOUT adjudication and one exact agreement."""

    if (
        expected_sha256 is not None
        and hashlib.sha256(adjudication_path.read_bytes()).hexdigest() != expected_sha256
    ):
        raise ValueError("HOLDOUT adjudication SHA-256 does not match the locked artifact")
    adjudication_value = json.loads(adjudication_path.read_text(encoding="utf-8"))
    if not isinstance(adjudication_value, dict):
        raise ValueError("HOLDOUT adjudication must be a JSON object")
    adjudication = cast(dict[str, object], adjudication_value)
    if adjudication.get("record_count") != 39 or adjudication.get("split") != "holdout":
        raise ValueError("HOLDOUT adjudication must contain exactly 39 records")
    if adjudication.get("selection_fingerprint") != PHASE11_SELECTION_FINGERPRINT:
        raise ValueError("HOLDOUT adjudication selection fingerprint is not locked")
    if adjudication.get("human_verified") is True:
        raise ValueError("HOLDOUT adjudication cannot set human_verified=true")
    if (
        adjudication.get("adjudication_provenance")
        != "ai_adjudicated_after_independent_second_review"
    ):
        raise ValueError("HOLDOUT adjudication provenance is invalid")
    first_sha = hashlib.sha256(first_review_path.read_bytes()).hexdigest()
    second_sha = hashlib.sha256(second_review_path.read_bytes()).hexdigest()
    if verify_review_hashes and (
        adjudication.get("review_1_sha256") != first_sha
        or first_sha != PHASE11_HOLDOUT_FIRST_REVIEW_SHA256
    ):
        raise ValueError("HOLDOUT first-review hash does not match adjudication")
    if verify_review_hashes and (
        adjudication.get("review_2_sha256") != second_sha
        or second_sha != PHASE11_HOLDOUT_SECOND_REVIEW_SHA256
    ):
        raise ValueError("HOLDOUT second-review hash does not match adjudication")
    if disagreement_path is not None and not disagreement_path.is_file() and verify_review_hashes:
        raise ValueError("HOLDOUT disagreement packet is unavailable")
    if disagreement_path is not None and disagreement_path.is_file():
        disagreement_sha = hashlib.sha256(disagreement_path.read_bytes()).hexdigest()
        if verify_review_hashes and disagreement_sha != PHASE11_HOLDOUT_DISAGREEMENT_SHA256:
            raise ValueError("HOLDOUT disagreement packet hash is not locked")
        if (
            verify_review_hashes
            and adjudication.get("source_disagreement_sha256") != disagreement_sha
        ):
            raise ValueError("HOLDOUT adjudication source disagreement hash does not match")

    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    selected_ids = {
        str(row["canonical_unit_id"])
        for row in cast(list[dict[str, object]], selection_payload["rows"])
        if str(row["split"]) == "holdout"
    }
    current_records = _load_records(
        "holdout", annotation_root=annotation_root, selection_manifest_path=selection_manifest_path
    )
    if (
        len(current_records) != 40
        or {record.canonical_unit_id for record in current_records} != selected_ids
    ):
        raise ValueError("protected HOLDOUT selection is not exactly 40 records")
    current_by_id = {record.canonical_unit_id: record for record in current_records}
    raw_adjudications = adjudication.get("records")
    if not isinstance(raw_adjudications, list):
        raise ValueError("HOLDOUT adjudication records are malformed")
    raw_adjudication_items = cast(list[object], raw_adjudications)
    if any(not isinstance(item, dict) for item in raw_adjudication_items):
        raise ValueError("HOLDOUT adjudication records are malformed")
    adjudications: list[dict[str, object]] = []
    for item in raw_adjudication_items:
        if isinstance(item, dict):
            adjudications.append(cast(dict[str, object], item))
    decision_counts = {
        "review_1": sum(item.get("decision") == "review_1" for item in adjudications),
        "custom": sum(item.get("decision") == "custom" for item in adjudications),
    }
    if decision_counts != {"review_1": 37, "custom": 2}:
        raise ValueError("HOLDOUT adjudication decisions are not the locked 37+2 summary")
    if any(
        item.get("adjudication_provenance") != "ai_adjudicated_after_independent_second_review"
        for item in adjudications
    ):
        raise ValueError("HOLDOUT adjudication record provenance is invalid")
    adjudicated_ids = [str(item.get("record_id")) for item in adjudications]
    if len(set(adjudicated_ids)) != 39 or set(adjudicated_ids) - selected_ids:
        raise ValueError("HOLDOUT adjudication contains duplicate or non-HOLDOUT IDs")
    exact_ids = selected_ids - set(adjudicated_ids)
    if len(exact_ids) != 1:
        raise ValueError("HOLDOUT adjudication must leave exactly one exact-agreement ID")

    first_records = _load_review_records(first_review_path, expected_split="holdout")
    second_records = _load_review_records(second_review_path, expected_split="holdout")
    first_by_id = {str(item.get("canonical_unit_id")): item for item in first_records}
    second_by_id = {str(item.get("canonical_unit_id")): item for item in second_records}
    if set(first_by_id) != selected_ids or set(second_by_id) != selected_ids:
        raise ValueError("independent HOLDOUT review IDs do not match protected selection")
    exact_id = next(iter(exact_ids))
    first_annotation = SemanticProposal.model_validate(first_by_id[exact_id]["human_annotations"])
    second_annotation = SemanticProposal.model_validate(second_by_id[exact_id]["human_annotations"])
    if first_annotation.model_dump(mode="json") != second_annotation.model_dump(mode="json"):
        raise ValueError("excluded HOLDOUT record is not an exact dual-AI agreement")

    expected_ids = selected_ids
    updates: list[tuple[Path, dict[str, object]]] = []
    for item in adjudications:
        record_id = str(item["record_id"])
        current = current_by_id[record_id]
        if item.get("human_verified") is True:
            raise ValueError(f"HOLDOUT adjudication human_verified=true: {record_id}")
        final_annotation = SemanticProposal.model_validate(item.get("final_annotation"))
        updated = current.model_copy(
            update={
                "human_annotations": final_annotation,
                "annotation_status": "reviewed",
                "annotation_provenance": "ai_adjudicated_after_independent_second_review",
                "human_verified": False,
            }
        )
        errors = validate_annotation_record(updated, expected_ids)
        if errors:
            raise ValueError(f"record {record_id} annotation rejected: {'; '.join(errors)}")
        updates.append(
            (
                _private_record_path(record_id, annotation_root),
                cast(dict[str, object], updated.model_dump(mode="json")),
            )
        )
    exact_current = current_by_id[exact_id]
    exact_updated = exact_current.model_copy(
        update={
            "human_annotations": first_annotation,
            "annotation_status": "dual_ai_agreed",
            "annotation_provenance": "dual_ai_agreed",
            "human_verified": False,
        }
    )
    errors = validate_annotation_record(exact_updated, expected_ids)
    if errors:
        raise ValueError(f"record {exact_id} annotation rejected: {'; '.join(errors)}")
    updates.append(
        (
            _private_record_path(exact_id, annotation_root),
            cast(dict[str, object], exact_updated.model_dump(mode="json")),
        )
    )
    annotation_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".phase11-holdout-adjudication-", dir=annotation_root
    ) as name:
        staging_root = Path(name)
        staged: list[tuple[Path, Path]] = []
        for target, record_payload in updates:
            staged_path = staging_root / target.name
            write_private_json(staged_path, record_payload)
            staged.append((staged_path, target))
        for staged_path, target in staged:
            os.replace(staged_path, target)
    return {
        "split": "holdout",
        "adjudicated_records": 39,
        "dual_ai_agreed_records": 1,
        "exact_agreement_id": exact_id,
        "human_verified": False,
        "holdout_records": 40,
    }


def _annotation_collection_counts(records: list[AnnotationRecord]) -> dict[str, int]:
    annotations = [
        record.human_annotations for record in records if record.human_annotations is not None
    ]
    return {
        "rules": sum(len(annotation.rules) for annotation in annotations),
        "regulated_entities": sum(len(annotation.regulated_entities) for annotation in annotations),
        "exceptions": sum(len(annotation.exceptions) for annotation in annotations),
        "penalties": sum(len(annotation.penalties) for annotation in annotations),
        "deadline_refs": sum(len(annotation.deadline_refs) for annotation in annotations),
        "effective_date_refs": sum(
            len(annotation.effective_date_refs) for annotation in annotations
        ),
        "monetary_threshold_refs": sum(
            len(annotation.monetary_threshold_refs) for annotation in annotations
        ),
        "percentage_threshold_refs": sum(
            len(annotation.percentage_threshold_refs) for annotation in annotations
        ),
        "empty_records": sum(
            not annotation.rules
            and not annotation.regulated_entities
            and not annotation.exceptions
            and not annotation.penalties
            and not annotation.deadline_refs
            and not annotation.effective_date_refs
            and not annotation.monetary_threshold_refs
            and not annotation.percentage_threshold_refs
            for annotation in annotations
        ),
    }


def freeze_holdout_annotation_release(
    *,
    annotation_root: Path = ANNOTATION_ROOT,
    selection_manifest_path: Path = SELECTION_MANIFEST_PATH,
    private_output_path: Path = HOLDOUT_RELEASE_PATH,
    tracked_output_path: Path = HOLDOUT_RELEASE_MANIFEST_PATH,
    adjudication_path: Path = HOLDOUT_ADJUDICATION_PATH,
) -> dict[str, object]:
    """Freeze the final AI-reviewed HOLDOUT reference without human-gold status."""

    if private_output_path.exists() or tracked_output_path.exists():
        raise ValueError("HOLDOUT annotation release is already frozen")
    records = _load_records(
        "holdout", annotation_root=annotation_root, selection_manifest_path=selection_manifest_path
    )
    if len(records) != 40:
        raise ValueError("HOLDOUT release requires exactly 40 records")
    errors = [
        error
        for record in records
        for error in validate_annotation_record(
            record, {item.canonical_unit_id for item in records}
        )
    ]
    if errors:
        raise ValueError("HOLDOUT annotations are invalid: " + "; ".join(errors))
    state_counts = {
        "ai_adjudicated_after_independent_second_review": sum(
            record.annotation_provenance == "ai_adjudicated_after_independent_second_review"
            for record in records
        ),
        "dual_ai_agreed": sum(
            record.annotation_provenance == "dual_ai_agreed" for record in records
        ),
        "independent_ai_review": sum(
            record.annotation_provenance == "independent_ai_review" for record in records
        ),
        "human_verified": sum(record.human_verified for record in records),
    }
    if state_counts != {
        "ai_adjudicated_after_independent_second_review": 39,
        "dual_ai_agreed": 1,
        "independent_ai_review": 0,
        "human_verified": 0,
    }:
        raise ValueError("HOLDOUT annotation provenance is not the final 39+1 state")
    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    source_batch_sha = hashlib.sha256(HOLDOUT_BATCH_PATH.read_bytes()).hexdigest()
    record_fingerprints: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.canonical_unit_id):
        annotation_json = json.dumps(
            record.human_annotations.model_dump(mode="json") if record.human_annotations else None,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        registry_json = json.dumps(
            record.candidate_registry.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        record_fingerprints.append(
            {
                "canonical_unit_id": record.canonical_unit_id,
                "annotation_provenance": record.annotation_provenance,
                "annotation_sha256": hashlib.sha256(annotation_json.encode("utf-8")).hexdigest(),
                "candidate_registry_sha256": hashlib.sha256(
                    registry_json.encode("utf-8")
                ).hexdigest(),
            }
        )
    release_fingerprint = hashlib.sha256(
        json.dumps(record_fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    candidate_fingerprint = _candidate_collection_fingerprint(records)
    aggregate = _annotation_collection_counts(records)
    adjudication_sha = hashlib.sha256(adjudication_path.read_bytes()).hexdigest()
    private_payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "phase11_holdout_annotation_release",
        "release_version": "phase11-holdout-annotation-release-v1",
        "lifecycle_state": "frozen",
        "status": "HOLDOUT_REFERENCE_FROZEN",
        "split": "holdout",
        "record_count": 40,
        "selection_fingerprint": selection_payload["selection_fingerprint"],
        "corpus_fingerprint": selection_payload.get("corpus_fingerprint"),
        "candidate_policy_version": CANDIDATE_REGISTRY_VERSION,
        "candidate_policy_hash": PHASE11_CANDIDATE_POLICY_HASH,
        "candidate_collection_fingerprint": candidate_fingerprint,
        "annotation_schema_version": "phase11-proposal-v1",
        "annotation_release_fingerprint": release_fingerprint,
        "record_fingerprints": record_fingerprints,
        "provenance_state_counts": state_counts,
        "aggregate_annotation_counts": aggregate,
        "human_verified": 0,
        "human_gold": False,
        "adjudication_artifact_sha256": adjudication_sha,
        "source_batch_sha256": source_batch_sha,
        "selection_manifest_sha256": hashlib.sha256(
            selection_manifest_path.read_bytes()
        ).hexdigest(),
        "evaluation_performed": 0,
    }
    write_private_json(private_output_path, private_payload)
    private_sha = hashlib.sha256(private_output_path.read_bytes()).hexdigest()
    tracked_payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "phase11_holdout_annotation_release",
        "release_version": "phase11-holdout-annotation-release-v1",
        "status": "HOLDOUT_REFERENCE_FROZEN",
        "split": "holdout",
        "record_count": 40,
        "provenance_state_counts": state_counts,
        "human_verified": 0,
        "human_gold": False,
        "selection_fingerprint": selection_payload["selection_fingerprint"],
        "corpus_fingerprint": selection_payload.get("corpus_fingerprint"),
        "candidate_policy_version": CANDIDATE_REGISTRY_VERSION,
        "candidate_policy_hash": PHASE11_CANDIDATE_POLICY_HASH,
        "candidate_collection_fingerprint": candidate_fingerprint,
        "annotation_schema_version": "phase11-proposal-v1",
        "annotation_release_fingerprint": release_fingerprint,
        "private_release_sha256": private_sha,
        "adjudication_artifact_sha256": adjudication_sha,
        "source_batch_sha256": source_batch_sha,
        "selection_manifest_sha256": hashlib.sha256(
            selection_manifest_path.read_bytes()
        ).hexdigest(),
        "aggregate_annotation_counts": aggregate,
        "evaluation_performed": 0,
    }
    write_text_free_json(tracked_output_path, tracked_payload)
    return {
        "path": private_output_path.as_posix(),
        "manifest_path": tracked_output_path.as_posix(),
        "sha256": private_sha,
        "manifest_sha256": hashlib.sha256(tracked_output_path.read_bytes()).hexdigest(),
        "annotation_release_fingerprint": release_fingerprint,
        "record_count": 40,
        "ai_adjudicated": state_counts["ai_adjudicated_after_independent_second_review"],
        **state_counts,
        **aggregate,
        "status": "HOLDOUT_REFERENCE_FROZEN",
    }
