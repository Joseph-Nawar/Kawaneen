# pyright: basic
"""Online retrieval latency summaries."""

from __future__ import annotations

import math
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    p50: float
    p95: float
    device: str
    package_versions: Mapping[str, str]
    threads: int

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[float],
        *,
        device: str,
        package_versions: Mapping[str, str],
        threads: int,
    ) -> LatencySummary:
        if not samples or any(value < 0 for value in samples):
            raise ValueError("latency samples must be non-empty and non-negative")
        return cls(
            count=len(samples),
            p50=_nearest_rank(samples, 0.50),
            p95=_nearest_rank(samples, 0.95),
            device=device,
            package_versions=dict(package_versions),
            threads=threads,
        )


def measure_latency(
    search: object,
    queries: Sequence[str],
    *,
    warmup_count: int = 3,
    device: str = "unknown",
    package_versions: Mapping[str, str] | None = None,
    threads: int = 1,
    observations: list[float] | None = None,
) -> LatencySummary:
    if not callable(search):
        raise TypeError("search must be callable")
    if not queries:
        raise ValueError("queries must be non-empty")
    warmups = min(max(warmup_count, 0), len(queries))
    for query in queries[:warmups]:
        search(query)
    samples = []
    for query in queries:
        started = time.perf_counter()
        search(query)
        samples.append((time.perf_counter() - started) * 1000.0)
    if observations is not None:
        observations.extend(samples)
    return LatencySummary.from_samples(
        samples,
        device=device if device != "unknown" else platform.machine(),
        package_versions=package_versions or {},
        threads=threads,
    )
