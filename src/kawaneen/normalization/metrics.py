"""Deterministic lexical retrieval metrics for the private Phase 4 ablation."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence


def query_metrics(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> dict[str, float]:
    relevant = set(relevant_ids)
    values: dict[str, float] = {}
    for cutoff in (1, 5, 10):
        returned = set(retrieved_ids[:cutoff])
        values[f"recall_at_{cutoff}"] = len(returned & relevant) / max(len(relevant), 1)
    reciprocal = 0.0
    for rank, unit_id in enumerate(retrieved_ids[:10], start=1):
        if unit_id in relevant:
            reciprocal = 1.0 / rank
            break
    values["mrr_at_10"] = reciprocal
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, unit_id in enumerate(retrieved_ids[:10])
        if unit_id in relevant
    )
    ideal_count = min(len(relevant), 10)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    values["ndcg_at_10"] = dcg / ideal if ideal else 0.0
    return values


def aggregate_query_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    values = tuple(rows)
    keys = sorted(values[0]) if values else []
    return {key: sum(row[key] for row in values) / max(len(values), 1) for key in keys}


def compare_query_metric(
    left: Sequence[dict[str, float]], right: Sequence[dict[str, float]], metric: str
) -> dict[str, int]:
    if len(left) != len(right):
        raise ValueError("paired metric rows must have equal length")
    wins = sum(a[metric] > b[metric] for a, b in zip(left, right, strict=True))
    losses = sum(a[metric] < b[metric] for a, b in zip(left, right, strict=True))
    return {"wins": wins, "ties": len(left) - wins - losses, "losses": losses}


def paired_confidence_interval(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int,
    replicates: int = 2000,
) -> dict[str, float | int]:
    if len(left) != len(right) or not left:
        raise ValueError("paired confidence intervals require equal non-empty samples")
    differences = tuple(a - b for a, b in zip(left, right, strict=True))
    estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    lower_index = int(0.025 * (len(samples) - 1))
    upper_index = int(0.975 * (len(samples) - 1))
    return {
        "estimate": estimate,
        "lower": samples[lower_index],
        "upper": samples[upper_index],
        "seed": seed,
        "replicates": replicates,
    }
