from __future__ import annotations

import json
from pathlib import Path

from kawaneen.generation.checkpoints import (
    GENERATION_CHECKPOINT_ARTIFACT_TYPE,
    GENERATION_CHECKPOINT_SCHEMA_VERSION,
    GenerationCheckpointStore,
    QueryCheckpoint,
)


def test_query_checkpoint_round_trip_and_resume_validation(tmp_path: Path) -> None:
    store = GenerationCheckpointStore(tmp_path)
    checkpoint = QueryCheckpoint(
        query_id="q1",
        generator_name="qwen-ollama",
        result_path="results/q1.json",
        fingerprint="a" * 64,
    )

    store.write(checkpoint)

    assert store.valid("q1", "a" * 64)
    assert not store.valid("q1", "b" * 64)


def test_corrupt_query_checkpoint_is_not_reused(tmp_path: Path) -> None:
    store = GenerationCheckpointStore(tmp_path)
    path = tmp_path / "q1.json"
    path.write_text("not-json", encoding="utf-8")

    assert not store.valid("q1", "a" * 64)


def test_checkpoint_fingerprint_changes_invalidate_resume(tmp_path: Path) -> None:
    store = GenerationCheckpointStore(tmp_path)
    store.write(
        QueryCheckpoint(
            query_id="q1",
            generator_name="qwen-ollama",
            result_path="results/q1.json",
            fingerprint="a" * 64,
        )
    )

    assert not store.valid("q1", "b" * 64)


def test_checkpoint_file_is_text_free_metadata(tmp_path: Path) -> None:
    store = GenerationCheckpointStore(tmp_path)
    store.write(
        QueryCheckpoint(
            query_id="q1",
            generator_name="qwen-ollama",
            result_path="results/q1.json",
            fingerprint="a" * 64,
        )
    )

    payload = json.loads((tmp_path / "q1.json").read_text(encoding="utf-8"))
    assert "quoted_text" not in payload
    assert payload["query_id"] == "q1"


def test_stage_c_legacy_checkpoint_is_not_complete(tmp_path: Path) -> None:
    store = GenerationCheckpointStore(tmp_path, require_complete_lifecycle=True)
    store.write(
        QueryCheckpoint(
            query_id="q1",
            generator_name="qwen-ollama-stage-c",
            result_path="results/q1.json",
            fingerprint="a" * 64,
        )
    )

    assert not store.valid("q1", "a" * 64)
    assert store.status() == {"completed": 0, "incomplete": 1, "corrupt": 0}


def test_stage_c_completed_generation_checkpoint_is_resumable(tmp_path: Path) -> None:
    result = tmp_path / "results" / "q1.json"
    result.parent.mkdir()
    result.write_text(
        json.dumps(
            {
                "artifact_type": "generation_result",
                "schema_version": GENERATION_CHECKPOINT_SCHEMA_VERSION,
                "lifecycle_state": "complete",
                "completion_kind": "generation",
                "query_id": "q1",
                "fingerprint": "a" * 64,
                "raw_output": "{}",
                "result": {"decision": "abstain"},
                "final_postprocessing_completed": True,
            }
        ),
        encoding="utf-8",
    )
    store = GenerationCheckpointStore(tmp_path / "checkpoints", require_complete_lifecycle=True)
    store.write(
        QueryCheckpoint(
            artifact_type=GENERATION_CHECKPOINT_ARTIFACT_TYPE,
            schema_version=GENERATION_CHECKPOINT_SCHEMA_VERSION,
            lifecycle_state="complete",
            completion_kind="generation",
            context_prepared=True,
            generation_attempted=True,
            generation_completed=True,
            final_postprocessing_completed=True,
            query_id="q1",
            generator_name="qwen-ollama-stage-c",
            result_path=result.as_posix(),
            fingerprint="a" * 64,
        )
    )

    assert store.valid("q1", "a" * 64)
    assert store.status() == {"completed": 1, "incomplete": 0, "corrupt": 0}


def test_stage_c_pre_generation_policy_checkpoint_is_resumable(tmp_path: Path) -> None:
    result = tmp_path / "results" / "q1.json"
    result.parent.mkdir()
    result.write_text(
        json.dumps(
            {
                "artifact_type": "generation_result",
                "schema_version": GENERATION_CHECKPOINT_SCHEMA_VERSION,
                "lifecycle_state": "complete",
                "completion_kind": "pre_generation_policy",
                "query_id": "q1",
                "fingerprint": "a" * 64,
                "raw_output": None,
                "result": {"decision": "abstain"},
                "final_postprocessing_completed": True,
                "pre_generation_policy_decision": "JURISDICTION_MISMATCH",
            }
        ),
        encoding="utf-8",
    )
    store = GenerationCheckpointStore(tmp_path / "checkpoints", require_complete_lifecycle=True)
    store.write(
        QueryCheckpoint(
            artifact_type=GENERATION_CHECKPOINT_ARTIFACT_TYPE,
            schema_version=GENERATION_CHECKPOINT_SCHEMA_VERSION,
            lifecycle_state="complete",
            completion_kind="pre_generation_policy",
            context_prepared=True,
            generation_attempted=False,
            generation_completed=False,
            final_postprocessing_completed=True,
            pre_generation_policy_decision="JURISDICTION_MISMATCH",
            query_id="q1",
            generator_name="qwen-ollama-stage-c",
            result_path=result.as_posix(),
            fingerprint="a" * 64,
        )
    )

    assert store.valid("q1", "a" * 64)
