"""Explicit lifecycle checkpoints for isolated extraction results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

from kawaneen.extraction.contracts import ExtractionModel


def extraction_fingerprint(**components: str) -> str:
    encoded = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ExtractionAttempt(ExtractionModel):
    attempt_number: int = Field(ge=1)
    lifecycle_state: Literal["incomplete", "complete", "failed"]
    failure_type: str | None = None
    retry_reason: str | None = None


class ExtractionCheckpoint(ExtractionModel):
    checkpoint_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    result_path: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_unit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    extractor_configuration: Literal["deterministic-v1", "hybrid-qwen-v1"]
    candidate_version: str = Field(min_length=1)
    prompt_hash: str = Field(pattern=r"^(none|[0-9a-f]{64})$")
    schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    qwen_model: str = Field(min_length=1)
    qwen_digest: str | None = None
    tokenizer_revision: str = Field(min_length=1)
    semantic_validation_policy: str = Field(min_length=1)
    lifecycle_state: Literal["incomplete", "complete", "failed"] = "incomplete"
    failure_reason: str | None = None
    failure_type: str | None = None
    attempt_count: int = Field(default=1, ge=1)
    prior_failure_type: str | None = None
    retry_reason: str | None = None
    attempt_history: tuple[ExtractionAttempt, ...] = ()
    context_prepared: bool = False
    extraction_attempted: bool = False
    extraction_completed: bool = False
    final_validation_completed: bool = False

    @model_validator(mode="after")
    def validate_completion(self) -> ExtractionCheckpoint:
        if self.lifecycle_state == "complete" and not (
            self.context_prepared
            and self.extraction_attempted
            and self.extraction_completed
            and self.final_validation_completed
        ):
            raise ValueError("complete checkpoint lacks lifecycle milestones")
        if (
            self.extractor_configuration == "hybrid-qwen-v1"
            and self.qwen_digest is None
            and self.lifecycle_state == "complete"
        ):
            raise ValueError("completed hybrid checkpoint requires an immutable Qwen digest")
        if self.lifecycle_state == "failed" and not self.failure_reason:
            raise ValueError("failed checkpoint requires a failure reason")
        return self


class ExtractionCheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, record_id: str) -> Path:
        if not record_id or Path(record_id).name != record_id or "/" in record_id:
            raise ValueError("unsafe extraction record ID")
        return self.root / f"{record_id}.json"

    def write(self, checkpoint: ExtractionCheckpoint) -> None:
        path = self.path_for(checkpoint.record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(checkpoint.model_dump(mode="json"), sort_keys=True, indent=2).encode(
            "utf-8"
        )
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def load(self, record_id: str) -> ExtractionCheckpoint:
        return ExtractionCheckpoint.model_validate(
            json.loads(self.path_for(record_id).read_text(encoding="utf-8"))
        )

    def valid(self, record_id: str, fingerprint: str) -> bool:
        try:
            checkpoint = self.load(record_id)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if checkpoint.fingerprint != fingerprint or checkpoint.lifecycle_state != "complete":
            return False
        try:
            payload_value: object = json.loads(
                Path(checkpoint.result_path).read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(payload_value, dict):
            return False
        payload = cast(dict[str, object], payload_value)
        return (
            payload.get("artifact_type") == "phase11_extraction_result"
            and payload.get("lifecycle_state") == "complete"
        )

    def status(self) -> dict[str, int]:
        completed = incomplete = failed = corrupt = 0
        for path in sorted(self.root.glob("*.json")) if self.root.is_dir() else ():
            try:
                checkpoint = ExtractionCheckpoint.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                if checkpoint.lifecycle_state == "complete" and self.valid(
                    checkpoint.record_id, checkpoint.fingerprint
                ):
                    completed += 1
                elif checkpoint.lifecycle_state == "failed":
                    failed += 1
                else:
                    incomplete += 1
            except (OSError, ValueError, json.JSONDecodeError):
                corrupt += 1
        return {
            "completed": completed,
            "incomplete": incomplete,
            "failed": failed,
            "corrupt": corrupt,
        }
