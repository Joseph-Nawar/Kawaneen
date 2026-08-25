"""Matched-80 generator governance and DEV-only outcome summaries."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import ALLAM_MODEL, GeneratorSubsetManifest, ModelLock, Phase15Model


class MatchedGeneratorOutput(Phase15Model):
    generator: str
    query_ids: tuple[str, ...]
    context_fingerprints: tuple[str, ...]
    outcomes: tuple[Mapping[str, object], ...]


def validate_matched_outputs(outputs: Sequence[MatchedGeneratorOutput], subset: GeneratorSubsetManifest) -> None:
    expected = tuple(
        subset.answerable_gold_present_ids
        + subset.answerable_gold_absent_ids
        + subset.unanswerable_ids
    )
    if len(expected) != 80 or len(set(expected)) != 80:
        raise ValueError("matched generator subset must contain exactly 80 unique IDs")
    if not outputs:
        raise ValueError("at least one generator output is required")
    first_ids = outputs[0].query_ids
    first_contexts = outputs[0].context_fingerprints
    if first_ids != expected:
        raise ValueError("generator output IDs do not match the frozen matched-80 order")
    if len(first_contexts) != 80:
        raise ValueError("each generator must use exactly 80 context blocks")
    for output in outputs:
        if output.query_ids != first_ids or output.context_fingerprints != first_contexts:
            raise ValueError("generator outputs must use identical IDs and context fingerprints")
        if len(output.outcomes) != 80:
            raise ValueError("each generator must have 80 outcomes")


def validate_allam_preflight(
    *,
    model_id: str,
    revision: str,
    quantization_bits: int,
    artifact_sha256: str | None,
    runtime: str,
    device: str,
    bounded_smoke_passed: bool,
    scoring_started: bool = False,
) -> dict[str, object]:
    if model_id != ALLAM_MODEL:
        raise ValueError(f"ALLaM preflight requires official model {ALLAM_MODEL}")
    if quantization_bits != 4:
        raise ValueError("full precision and non-4-bit ALLaM paths are forbidden")
    if not revision or len(revision) != 40:
        raise ValueError("ALLaM preflight requires an exact official revision")
    if not artifact_sha256:
        raise ValueError("ALLaM preflight requires a provenance-linked artifact SHA")
    if scoring_started and not bounded_smoke_passed:
        raise ValueError("ALLaM scoring is forbidden before bounded preflight")
    return {
        "model_id": model_id,
        "revision": revision,
        "quantization_bits": quantization_bits,
        "artifact_sha256": artifact_sha256,
        "runtime": runtime,
        "device": device,
        "bounded_smoke_passed": bounded_smoke_passed,
        "full_precision_forbidden": True,
        "scoring_permitted": bounded_smoke_passed,
    }


def summarize_generator_outputs(outputs: Sequence[MatchedGeneratorOutput]) -> dict[str, object]:
    summary: dict[str, object] = {"generators": {}}
    generators: dict[str, object] = {}
    for output in outputs:
        counts = Counter(str(item.get("outcome", "invalid generation")) for item in output.outcomes)
        generators[output.generator] = {
            "n": len(output.outcomes),
            "outcome_taxonomy": dict(sorted(counts.items())),
            "successful_verified_answer_count": sum(
                bool(item.get("verified_answer")) for item in output.outcomes
            ),
        }
    summary["generators"] = generators
    return summary


def model_lock_from_preflight(preflight: Mapping[str, object], *, context_limit: int, output_limit: int, disk_footprint_bytes: int) -> ModelLock:
    return ModelLock(
        model_id=str(preflight["model_id"]),
        revision=str(preflight["revision"]),
        dtype="4-bit",
        batch_size=1,
        runtime=str(preflight["runtime"]),
        device=str(preflight["device"]),
        quantization={
            "bits": 4,
            "artifact_sha256": str(preflight["artifact_sha256"]),
            "context_limit": context_limit,
            "output_limit": output_limit,
            "disk_footprint_bytes": disk_footprint_bytes,
        },
        preflight=dict(preflight),
    )
