"""Text-free deterministic checkpoint manifests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

GENERATION_CHECKPOINT_ARTIFACT_TYPE = "generation_checkpoint"
GENERATION_RESULT_ARTIFACT_TYPE = "generation_result"
GENERATION_CHECKPOINT_SCHEMA_VERSION = 2


class CheckpointManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_id: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    generator_name: str = Field(min_length=1)
    completed_query_ids: tuple[str, ...] = ()
    artifact_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class QueryCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1)
    generator_name: str = Field(min_length=1)
    result_path: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    telemetry: dict[str, object] = {}
    artifact_type: str = GENERATION_CHECKPOINT_ARTIFACT_TYPE
    schema_version: int = 1
    lifecycle_state: str = "legacy"
    completion_kind: Literal["generation", "pre_generation_policy"] | None = None
    context_prepared: bool = False
    generation_attempted: bool = False
    generation_completed: bool = False
    final_postprocessing_completed: bool = False
    pre_generation_policy_decision: str | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> QueryCheckpoint:
        if self.schema_version == GENERATION_CHECKPOINT_SCHEMA_VERSION:
            if self.artifact_type != GENERATION_CHECKPOINT_ARTIFACT_TYPE:
                raise ValueError("invalid generation checkpoint artifact type")
            if self.lifecycle_state == "complete":
                if not self.context_prepared or not self.final_postprocessing_completed:
                    raise ValueError("complete checkpoint lacks lifecycle milestones")
                if self.completion_kind == "generation":
                    if not self.generation_attempted or not self.generation_completed:
                        raise ValueError(
                            "complete generation checkpoint lacks generation milestone"
                        )
                elif self.completion_kind == "pre_generation_policy":
                    if self.generation_attempted or self.generation_completed:
                        raise ValueError("policy checkpoint cannot claim generation completion")
                    if not self.pre_generation_policy_decision:
                        raise ValueError("policy checkpoint lacks explicit policy decision")
                else:
                    raise ValueError("complete checkpoint lacks completion kind")
            elif self.lifecycle_state not in {"incomplete", "legacy"}:
                raise ValueError("unknown generation checkpoint lifecycle state")
        return self

    def is_complete_stage_c(self) -> bool:
        return (
            self.artifact_type == GENERATION_CHECKPOINT_ARTIFACT_TYPE
            and self.schema_version == GENERATION_CHECKPOINT_SCHEMA_VERSION
            and self.lifecycle_state == "complete"
            and self.completion_kind in {"generation", "pre_generation_policy"}
        )


class GenerationCheckpointStore:
    """Atomic private per-query checkpoints for resumable generation."""

    def __init__(self, root: Path, *, require_complete_lifecycle: bool = False) -> None:
        self.root = root
        self.require_complete_lifecycle = require_complete_lifecycle

    def path_for(self, query_id: str) -> Path:
        if not query_id or Path(query_id).name != query_id or "/" in query_id:
            raise ValueError("unsafe generation checkpoint query ID")
        return self.root / f"{query_id}.json"

    def write(self, checkpoint: QueryCheckpoint) -> None:
        path = self.path_for(checkpoint.query_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            checkpoint.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def load(self, query_id: str) -> QueryCheckpoint:
        path = self.path_for(query_id)
        return QueryCheckpoint.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def valid(self, query_id: str, fingerprint: str) -> bool:
        try:
            checkpoint = self.load(query_id)
        except (OSError, json.JSONDecodeError, ValueError):
            return False
        if checkpoint.query_id != query_id or checkpoint.fingerprint != fingerprint:
            return False
        if not self.require_complete_lifecycle:
            return True
        if not checkpoint.is_complete_stage_c():
            return False
        return _result_proves_completion(checkpoint)

    def status(self) -> dict[str, int]:
        completed = 0
        incomplete = 0
        corrupt = 0
        for path in sorted(self.root.glob("*.json")) if self.root.is_dir() else ():
            try:
                checkpoint = QueryCheckpoint.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                if self.require_complete_lifecycle:
                    if checkpoint.is_complete_stage_c() and _result_proves_completion(checkpoint):
                        completed += 1
                    else:
                        incomplete += 1
                else:
                    completed += 1
            except (OSError, json.JSONDecodeError, ValueError):
                corrupt += 1
        if self.require_complete_lifecycle:
            return {"completed": completed, "incomplete": incomplete, "corrupt": corrupt}
        return {"completed": completed, "corrupt": corrupt}


def _result_proves_completion(checkpoint: QueryCheckpoint) -> bool:
    try:
        result_path = Path(checkpoint.result_path)
        payload_value = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload_value, dict):
            return False
        payload = cast(dict[str, object], payload_value)
        if (
            payload.get("artifact_type") != GENERATION_RESULT_ARTIFACT_TYPE
            or payload.get("schema_version") != GENERATION_CHECKPOINT_SCHEMA_VERSION
            or payload.get("lifecycle_state") != "complete"
            or payload.get("completion_kind") != checkpoint.completion_kind
            or payload.get("query_id") != checkpoint.query_id
            or payload.get("fingerprint") != checkpoint.fingerprint
            or payload.get("final_postprocessing_completed") is not True
            or not isinstance(payload.get("result"), dict)
        ):
            return False
        result = cast(dict[str, object], payload["result"])
        if checkpoint.completion_kind == "generation":
            return isinstance(payload.get("raw_output"), str) and result.get("decision") in {
                "answer",
                "abstain",
            }
        return (
            checkpoint.pre_generation_policy_decision is not None
            and payload.get("pre_generation_policy_decision")
            == checkpoint.pre_generation_policy_decision
            and result.get("decision") == "abstain"
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def write_checkpoint(path: Path, manifest: CheckpointManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path) -> CheckpointManifest:
    return CheckpointManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
