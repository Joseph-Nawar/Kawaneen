"""Preparation-only experiment orchestration; execution is intentionally later."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=1)
    generator_names: tuple[str, ...] = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    generator_names: tuple[str, ...]
    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_allowed: bool = False


def prepare_experiment(spec: ExperimentSpec) -> ExperimentPlan:
    encoded = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return ExperimentPlan(
        experiment_id=spec.experiment_id,
        generator_names=spec.generator_names,
        plan_fingerprint=fingerprint,
    )
