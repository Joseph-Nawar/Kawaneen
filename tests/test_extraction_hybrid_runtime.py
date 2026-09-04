from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.annotation import AnnotationRecord
from kawaneen.extraction.artifacts import write_private_json
from kawaneen.extraction.candidates import build_candidate_registry
from kawaneen.extraction.hybrid_prompt import (
    HYBRID_QWEN_HF_ID,
    HYBRID_QWEN_HF_REVISION,
    HYBRID_QWEN_MODEL,
    HYBRID_QWEN_OLLAMA_DIGEST,
    HYBRID_QWEN_TOKENIZER_REVISION,
    hybrid_prompt_hash,
    hybrid_schema_hash,
    render_hybrid_prompt,
)
from kawaneen.extraction.orchestration import (
    _load_holdout_source_records_for_inference,
    run_hybrid_records,
    run_hybrid_split,
)
from kawaneen.extraction.provider import MockExtractionProvider


def test_ollama_extraction_provider_accepts_compose_service_hostname() -> None:
    from kawaneen.extraction.provider import OllamaExtractionProvider

    provider = OllamaExtractionProvider(
        endpoint="http://ollama:11434/api/generate",
        local_lock_path=Path("missing-lock.json"),
    )

    assert provider.identity_endpoint == "http://ollama:11434"


class _TimeoutProvider:
    def __init__(self) -> None:
        self.calls = 0

    def propose(self, canonical_text: str, registry: object) -> object:
        del canonical_text, registry
        self.calls += 1
        raise TimeoutError("timed out")


class _RuntimeFailureProvider:
    def __init__(self) -> None:
        self.calls = 0

    def propose(self, canonical_text: str, registry: object) -> object:
        del canonical_text, registry
        self.calls += 1
        raise RuntimeError("not a timeout")


def _record(record_id: str = "u1") -> AnnotationRecord:
    text = "يجب على المنشأة التسجيل خلال 30 يوماً."
    return AnnotationRecord(
        canonical_unit_id=record_id,
        document_id="d1",
        canonical_text=text,
        source_provenance=SourceProvenance(
            source_id="saudi-moj-derived",
            source_version="v3",
            source_path="private",
            source_row=1,
            source_field="text",
        ),
        source_fingerprint="a" * 64,
        split="dev",
        candidate_registry=build_candidate_registry(
            text,
            canonical_unit_id=record_id,
            document_id="d1",
        ),
    )


def _proposal() -> dict[str, object]:
    return {
        "schema_version": "phase11-proposal-v1",
        "regulated_entities": [{"text": "المنشأة"}],
        "rules": [
            {
                "modality": "obligation",
                "actor": {"text": "المنشأة"},
                "action": {"text": "التسجيل"},
                "deadline_refs": ["T001"],
            }
        ],
        "deadline_refs": ["T001"],
    }


def test_hybrid_prompt_and_schema_hashes_are_deterministic() -> None:
    record = _record()
    first = render_hybrid_prompt(record.canonical_text, record.candidate_registry)
    second = render_hybrid_prompt(record.canonical_text, record.candidate_registry)
    assert first == second
    assert first.text.count("يجب على المنشأة التسجيل") == 1
    assert "T001" in first.text
    assert hybrid_prompt_hash() == hybrid_prompt_hash()
    assert len(hybrid_schema_hash()) == 64
    assert HYBRID_QWEN_MODEL == "qwen3:4b-instruct-2507-q4_K_M"
    assert HYBRID_QWEN_HF_ID == "Qwen/Qwen3-4B-Instruct-2507"
    assert len(HYBRID_QWEN_HF_REVISION) == 40
    assert HYBRID_QWEN_OLLAMA_DIGEST.startswith("sha256:")
    assert len(HYBRID_QWEN_TOKENIZER_REVISION) == 40


def test_mock_hybrid_run_persists_complete_and_resume_skips_completed(tmp_path: Path) -> None:
    provider = MockExtractionProvider(_proposal())
    result = run_hybrid_records(
        [_record()],
        provider,
        checkpoint_root=tmp_path / "checkpoints",
        result_root=tmp_path / "results",
        selection_fingerprint="b" * 64,
        semantic_release_fingerprint="c" * 64,
        candidate_compatible_release_fingerprint="d" * 64,
        prompt_hash=hybrid_prompt_hash(),
        schema_hash=hybrid_schema_hash(),
        qwen_model=HYBRID_QWEN_MODEL,
        qwen_digest=HYBRID_QWEN_OLLAMA_DIGEST,
        tokenizer_revision=HYBRID_QWEN_TOKENIZER_REVISION,
    )
    assert result["completed"] == 1
    assert provider.calls == 1
    resumed = run_hybrid_records(
        [_record()],
        provider,
        checkpoint_root=tmp_path / "checkpoints",
        result_root=tmp_path / "results",
        selection_fingerprint="b" * 64,
        semantic_release_fingerprint="c" * 64,
        candidate_compatible_release_fingerprint="d" * 64,
        prompt_hash=hybrid_prompt_hash(),
        schema_hash=hybrid_schema_hash(),
        qwen_model=HYBRID_QWEN_MODEL,
        qwen_digest=HYBRID_QWEN_OLLAMA_DIGEST,
        tokenizer_revision=HYBRID_QWEN_TOKENIZER_REVISION,
        resume=True,
    )
    assert resumed["skipped"] == 1
    assert provider.calls == 1


def test_invalid_provider_output_is_failed_and_does_not_block_next_record(tmp_path: Path) -> None:
    provider = MockExtractionProvider(
        json.dumps(
            {
                "schema_version": "phase11-proposal-v1",
                "rules": [{"modality": "obligation", "action": {"text": "غير موجود"}}],
            },
            ensure_ascii=False,
        )
    )
    result = run_hybrid_records(
        [_record("u1"), _record("u2")],
        provider,
        checkpoint_root=tmp_path / "checkpoints",
        result_root=tmp_path / "results",
        selection_fingerprint="b" * 64,
        semantic_release_fingerprint="c" * 64,
        candidate_compatible_release_fingerprint="d" * 64,
        prompt_hash=hybrid_prompt_hash(),
        schema_hash=hybrid_schema_hash(),
        qwen_model=HYBRID_QWEN_MODEL,
        qwen_digest=HYBRID_QWEN_OLLAMA_DIGEST,
        tokenizer_revision=HYBRID_QWEN_TOKENIZER_REVISION,
    )
    assert result["failed"] == 2
    assert provider.calls == 2


def test_resume_keeps_failed_default_but_explicitly_retries_one_timeout_and_preserves_history(
    tmp_path: Path,
) -> None:
    timeout_provider = _TimeoutProvider()
    roots = {"checkpoint_root": tmp_path / "checkpoints", "result_root": tmp_path / "results"}
    first = run_hybrid_records(
        [_record()],
        timeout_provider,
        **roots,
        selection_fingerprint="b" * 64,
        semantic_release_fingerprint="c" * 64,
        candidate_compatible_release_fingerprint="d" * 64,
        prompt_hash=hybrid_prompt_hash(),
        schema_hash=hybrid_schema_hash(),
        qwen_model=HYBRID_QWEN_MODEL,
        qwen_digest=HYBRID_QWEN_OLLAMA_DIGEST,
        tokenizer_revision=HYBRID_QWEN_TOKENIZER_REVISION,
    )
    assert first["failed"] == 1

    default_resume_provider = MockExtractionProvider(_proposal())
    default_resume = run_hybrid_records(
        [_record()],
        default_resume_provider,
        **roots,
        selection_fingerprint="b" * 64,
        semantic_release_fingerprint="c" * 64,
        candidate_compatible_release_fingerprint="d" * 64,
        prompt_hash=hybrid_prompt_hash(),
        schema_hash=hybrid_schema_hash(),
        qwen_model=HYBRID_QWEN_MODEL,
        qwen_digest=HYBRID_QWEN_OLLAMA_DIGEST,
        tokenizer_revision=HYBRID_QWEN_TOKENIZER_REVISION,
        resume=True,
    )
    assert default_resume["skipped"] == 1
    assert default_resume_provider.calls == 0

    retry_provider = MockExtractionProvider(_proposal())
    retried = run_hybrid_records(
        [_record()],
        retry_provider,
        **roots,
        selection_fingerprint="b" * 64,
        semantic_release_fingerprint="c" * 64,
        candidate_compatible_release_fingerprint="d" * 64,
        prompt_hash=hybrid_prompt_hash(),
        schema_hash=hybrid_schema_hash(),
        qwen_model=HYBRID_QWEN_MODEL,
        qwen_digest=HYBRID_QWEN_OLLAMA_DIGEST,
        tokenizer_revision=HYBRID_QWEN_TOKENIZER_REVISION,
        resume=True,
        retry_timeouts=True,
        accept_field_local_diagnostics=True,
    )
    assert retried["completed"] == 1
    assert retried["provider_calls_attempted"] == 1
    assert retry_provider.calls == 1
    checkpoint = json.loads((tmp_path / "checkpoints" / "u1.json").read_text(encoding="utf-8"))
    assert checkpoint["attempt_count"] == 2
    assert checkpoint["prior_failure_type"] == "MODEL_TIMEOUT"
    assert checkpoint["retry_reason"] == "authorized_timeout_retry"
    assert [item["failure_type"] for item in checkpoint["attempt_history"]] == [
        "MODEL_TIMEOUT",
        None,
    ]
    attempt_one = list((tmp_path / "results" / "attempts" / "u1").glob("attempt-1.json"))
    assert len(attempt_one) == 1
    assert json.loads(attempt_one[0].read_text(encoding="utf-8"))["failure_type"] == (
        "MODEL_TIMEOUT"
    )


def test_timeout_retry_does_not_retry_non_timeout_failure(tmp_path: Path) -> None:
    provider = _RuntimeFailureProvider()
    kwargs = dict(
        checkpoint_root=tmp_path / "checkpoints",
        result_root=tmp_path / "results",
        selection_fingerprint="b" * 64,
        semantic_release_fingerprint="c" * 64,
        candidate_compatible_release_fingerprint="d" * 64,
        prompt_hash=hybrid_prompt_hash(),
        schema_hash=hybrid_schema_hash(),
        qwen_model=HYBRID_QWEN_MODEL,
        qwen_digest=HYBRID_QWEN_OLLAMA_DIGEST,
        tokenizer_revision=HYBRID_QWEN_TOKENIZER_REVISION,
    )
    run_hybrid_records([_record()], provider, **kwargs)
    retry_provider = MockExtractionProvider(_proposal())
    retried = run_hybrid_records(
        [_record()], retry_provider, resume=True, retry_timeouts=True, **kwargs
    )
    assert retried["skipped"] == 1
    assert retried["statuses"] == [{"record_id": "u1", "status": "skipped_failed"}]
    assert retry_provider.calls == 0


def test_hybrid_command_is_dev_only() -> None:
    with pytest.raises(ValueError, match="protected HOLDOUT"):
        run_hybrid_split("holdout")
    with pytest.raises(ValueError, match="cannot be used for DEV"):
        run_hybrid_split("dev", allow_holdout=True)


def test_holdout_inference_source_records_exclude_reference_annotations(
    tmp_path: Path,
) -> None:
    source_record = _record().model_copy(
        update={
            "split": "holdout",
            "human_annotations": _proposal(),
            "annotation_status": "reviewed",
            "annotation_provenance": "independent_ai_review",
        }
    )
    batch = tmp_path / "holdout-source.json"
    write_private_json(
        tmp_path / "selection.json",
        {
            "rows": [
                {
                    "canonical_unit_id": source_record.canonical_unit_id,
                    "split": "holdout",
                }
            ],
            "selection_fingerprint": "selection",
        },
    )
    write_private_json(
        batch,
        {
            "schema_version": "phase11-holdout-annotation-batch-v1",
            "split": "holdout",
            "selection_version": "phase11-selection-v2",
            "candidate_registry_version": "phase11-candidates-v3",
            "selection_fingerprint": "selection",
            "records": [
                source_record.model_copy(
                    update={
                        "human_annotations": None,
                        "annotation_status": "unreviewed",
                        "annotation_provenance": "unreviewed",
                        "human_verified": False,
                    }
                ).model_dump(mode="json")
            ],
        },
    )

    records = _load_holdout_source_records_for_inference(
        batch_path=batch,
        selection_manifest_path=tmp_path / "selection.json",
    )
    assert len(records) == 1
    assert records[0].human_annotations is None
    assert records[0].annotation_provenance == "unreviewed"
