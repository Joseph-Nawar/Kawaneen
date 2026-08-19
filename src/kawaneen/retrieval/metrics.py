"""Small pure deterministic information-retrieval metrics."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _relevant(retrieved: Sequence[str], qrels: Mapping[str, int], k: int) -> int:
    return sum(qrels.get(chunk_id, 0) > 0 for chunk_id in retrieved[:k])


def recall_at_k(retrieved: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    total = sum(grade > 0 for grade in qrels.values())
    return _relevant(retrieved, qrels, k) / total if total else 0.0


def precision_at_k(retrieved: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    return _relevant(retrieved, qrels, k) / k


def mrr_at_k(retrieved: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if qrels.get(chunk_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    def dcg(ids: Sequence[str]) -> float:
        return sum(
            (2 ** qrels.get(chunk_id, 0) - 1) / math.log2(rank + 2)
            for rank, chunk_id in enumerate(ids[:k])
        )

    ideal = dcg(tuple(sorted(qrels, key=lambda chunk_id: (-qrels[chunk_id], chunk_id))))
    return dcg(retrieved) / ideal if ideal else 0.0


def complete_evidence_recall_at_k(
    retrieved: Sequence[str], groups: Mapping[str, frozenset[str]], k: int
) -> float:
    if not groups:
        return 0.0
    hits = set(retrieved[:k])
    return float(all(hits & required for required in groups.values()))


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    low: float
    high: float
    replicates: int
    seed: int


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int,
    replicates: int = 2000,
    confidence: float = 0.95,
) -> BootstrapInterval:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equal non-empty samples")
    if replicates < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    samples = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(replicates)
    )
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, math.ceil(alpha * replicates) - 1)
    high_index = min(replicates - 1, math.ceil((1.0 - alpha) * replicates) - 1)
    return BootstrapInterval(
        estimate=estimate,
        low=samples[low_index],
        high=samples[high_index],
        replicates=replicates,
        seed=seed,
    )


def wins_ties_losses(left: Sequence[float], right: Sequence[float]) -> dict[str, int]:
    if len(left) != len(right):
        raise ValueError("paired comparison requires equal samples")
    return {
        "wins": sum(a > b for a, b in zip(left, right, strict=True)),
        "ties": sum(a == b for a, b in zip(left, right, strict=True)),
        "losses": sum(a < b for a, b in zip(left, right, strict=True)),
    }
