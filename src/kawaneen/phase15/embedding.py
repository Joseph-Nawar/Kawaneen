"""DEV-only embedding comparison contracts and aggregate metric helpers."""

from __future__ import annotations

from enum import StrEnum
from statistics import median
from typing import Mapping, Sequence

from .contracts import ARABIC_EMBEDDING_MODEL, ModelLock, Phase15Model
from .statistics import paired_bootstrap_delta


class NormalizationVariant(StrEnum):
    RAW = "arabic-raw-v1"
    LIGHT = "arabic-light-v1"
    AGGRESSIVE = "arabic-aggressive-v1"


class EmbeddingRun(Phase15Model):
    system: str
    normalization: str
    query_ids: tuple[str, ...]
    qrel_ids: tuple[str, ...]
    metrics: Mapping[str, tuple[float, ...]]
    latencies_ms: tuple[float, ...] = ()

    def validate_identity(self, expected_query_ids: Sequence[str], expected_qrel_ids: Sequence[str]) -> None:
        if self.query_ids != tuple(expected_query_ids) or self.qrel_ids != tuple(expected_qrel_ids):
            raise ValueError("embedding runs must use identical DEV query and qrel identities")
        if len(self.query_ids) != len(self.qrel_ids):
            raise ValueError("query and qrel identity lengths must match")


def create_arabic_model_lock(
    revision: str,
    *,
    runtime: str = "sentence-transformers",
    device: str = "cpu",
    batch_size: int = 1,
    dtype: str = "float32",
) -> ModelLock:
    if not revision or len(revision) != 40:
        raise ValueError("Arabic embedding lock requires a 40-character immutable revision")
    return ModelLock(
        model_id=ARABIC_EMBEDDING_MODEL,
        revision=revision,
        pooling="mean_tokens",
        dimension=768,
        normalize_embeddings=False,
        query_prefix="",
        passage_prefix="",
        dtype=dtype,
        batch_size=batch_size,
        runtime=runtime,
        device=device,
    )


def resolve_hf_revision(model_id: str) -> str:
    """Resolve public metadata only; this function never downloads model weights."""

    if model_id != ARABIC_EMBEDDING_MODEL:
        raise ValueError(f"unexpected Phase 15 Arabic model: {model_id}")
    from huggingface_hub import HfApi

    revision = HfApi().model_info(model_id).sha
    if not revision:
        raise RuntimeError(f"Hugging Face did not return an immutable revision for {model_id}")
    return revision


def evaluate_embedding_runs(
    runs: Sequence[EmbeddingRun],
    *,
    seed: int = 20260826,
    replicates: int = 2000,
) -> dict[str, object]:
    if not runs:
        raise ValueError("at least one embedding run is required")
    expected_queries, expected_qrels = runs[0].query_ids, runs[0].qrel_ids
    for run in runs:
        run.validate_identity(expected_queries, expected_qrels)
    output: dict[str, object] = {"systems": {}, "seed": seed, "bootstrap_replicates": replicates}
    systems: dict[str, object] = {}
    for run in runs:
        metric_values: dict[str, object] = {}
        for metric, values in run.metrics.items():
            if len(values) != len(expected_queries):
                raise ValueError(f"metric {metric} length does not match query identities")
            metric_values[metric] = {
                "mean": sum(values) / len(values),
                "n": len(values),
            }
        metric_values["p50_latency_ms"] = median(run.latencies_ms) if run.latencies_ms else None
        metric_values["p95_latency_ms"] = (
            sorted(run.latencies_ms)[max(0, int(len(run.latencies_ms) * 0.95) - 1)]
            if run.latencies_ms
            else None
        )
        systems[run.system] = metric_values
    output["systems"] = systems
    return output


def paired_embedding_delta(
    left: Sequence[float], right: Sequence[float], *, seed: int = 20260826
) -> dict[str, object]:
    return paired_bootstrap_delta(left, right, seed=seed).__dict__
