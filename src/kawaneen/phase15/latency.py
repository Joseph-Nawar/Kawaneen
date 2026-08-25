"""Batch-1 warmup and percentile latency measurement helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median
from time import monotonic_ns


@dataclass(frozen=True)
class LatencySummary:
    p50_ms: float
    p95_ms: float
    samples_ms: tuple[float, ...]
    warmups: int
    batch_size: int


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * probability)))
    return ordered[index]


def measure_latency(
    operation: Callable[[], object],
    *,
    samples: int = 10,
    warmups: int = 3,
    clock: Callable[[], int] = monotonic_ns,
) -> LatencySummary:
    if samples <= 0 or warmups < 3:
        raise ValueError("latency protocol requires positive samples and at least three warmups")
    for _ in range(warmups):
        operation()
    measurements: list[float] = []
    for _ in range(samples):
        start = clock()
        operation()
        measurements.append((clock() - start) / 1_000_000)
    return LatencySummary(
        p50_ms=median(measurements),
        p95_ms=_percentile(measurements, 0.95),
        samples_ms=tuple(measurements),
        warmups=warmups,
        batch_size=1,
    )
