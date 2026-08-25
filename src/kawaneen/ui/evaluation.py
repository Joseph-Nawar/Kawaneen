"""Sanitized tracked evaluation evidence and current-session latency helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median, quantiles
from typing import Any

_ALLOWED_SOURCES = (
    Path("data/evaluation/phase8_comparison.json"),
    Path("data/evaluation/phase10_qwen_stage_d_metrics.json"),
    Path("data/evaluation/phase11_hybrid_qwen_stage_b2_clean_holdout_report.json"),
)


@dataclass(frozen=True)
class EvaluationSource:
    path: str
    sha256: str


@dataclass(frozen=True)
class EvaluationSnapshot:
    schema_version: str
    sources: tuple[EvaluationSource, ...]
    retrieval: dict[str, Any]
    generation: dict[str, float]
    extraction: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatencySummary:
    values: tuple[float, ...]
    count: int
    median: float
    p95: float
    minimum: float
    maximum: float


def validate_source_path(root: Path, relative_path: Path) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("Evaluation source must remain inside the repository.") from error
    if relative_path not in _ALLOWED_SOURCES:
        raise ValueError("Evaluation source is not on the Phase 13 allowlist.")
    if "private" in candidate.parts:
        raise ValueError("Private evaluation sources are not allowed.")
    return candidate


def build_evaluation_snapshot(root: Path) -> EvaluationSnapshot:
    source_data: dict[Path, dict[str, Any]] = {}
    sources: list[EvaluationSource] = []
    for relative_path in _ALLOWED_SOURCES:
        path = validate_source_path(root, relative_path)
        raw = path.read_bytes()
        source_data[relative_path] = json.loads(raw)
        sources.append(EvaluationSource(str(relative_path), hashlib.sha256(raw).hexdigest()))

    phase8 = source_data[Path("data/evaluation/phase8_comparison.json")]
    phase10 = source_data[Path("data/evaluation/phase10_qwen_stage_d_metrics.json")]
    phase11 = source_data[
        Path("data/evaluation/phase11_hybrid_qwen_stage_b2_clean_holdout_report.json")
    ]
    end_to_end = phase11["end_to_end_40_record_view"]
    micro = end_to_end["micro"]
    safety = phase11["safety_structural_metrics"]
    retrieval = {
        "dev": _retrieval_models(phase8["dev_metrics"]),
        "holdout": _retrieval_models(phase8["holdout_metrics"]),
        "selected_deltas": phase8["deltas"],
        "scope_note": "Common tracked Phase 8 metrics; DEV and holdout are split labels.",
    }
    generation = {
        name: float(phase10["metrics"][name]["value"])
        for name in (
            "ValidCitationRate",
            "ClaimCitationCoverage",
            "final_answer_coverage",
            "FalseAnswerRate",
            "invalid_generation_rate",
        )
    }
    extraction = {
        "completion": float(safety["PipelineCompletionRate"]["rate"]),
        "micro_precision": float(micro["precision"]),
        "micro_recall": float(micro["recall"]),
        "micro_f1": float(micro["F1"]),
        "schema_validity": float(safety["FinalSchemaValidityRate"]["end_to_end"]["rate"]),
        "provenance_completeness": float(
            safety["ProvenanceCompletenessRate"]["end_to_end"]["rate"]
        ),
        "full_rule_exact_f1": float(end_to_end["normative"]["full_rule_exact_match"]["F1"]),
        "error_taxonomy": {
            key: int(value) for key, value in sorted(phase11["error_taxonomy"].items())
        },
        "scope_note": "Phase 11 protected HOLDOUT summary from the tracked sanitized report.",
    }
    return EvaluationSnapshot(
        "phase13-evaluation-v1", tuple(sources), retrieval, generation, extraction
    )


def write_evaluation_snapshot(root: Path, destination: Path) -> None:
    snapshot = build_evaluation_snapshot(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def aggregate_latency(values: Sequence[float], limit: int = 50) -> LatencySummary:
    recent = tuple(float(value) for value in values[-limit:])
    if not recent:
        return LatencySummary((), 0, 0.0, 0.0, 0.0, 0.0)
    return LatencySummary(
        values=recent,
        count=len(recent),
        median=float(median(recent)),
        p95=float(quantiles(recent, n=100, method="inclusive")[94])
        if len(recent) > 1
        else next(iter(recent)),
        minimum=min(recent),
        maximum=max(recent),
    )


def _retrieval_models(split: dict[str, Any]) -> dict[str, dict[str, float]]:
    models: dict[str, dict[str, float]] = {}
    for name, payload in split.items():
        metrics = payload.get("metrics", {})
        selected = {
            metric: float(
                metrics[metric]["value"]
                if isinstance(metrics[metric], dict) and "value" in metrics[metric]
                else metrics[metric]
            )
            for metric in ("MRR@10", "nDCG@10", "Recall@10")
            if metric in metrics
        }
        models[name] = selected
    return models
