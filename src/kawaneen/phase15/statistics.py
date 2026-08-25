"""Deterministic paired statistics used by Phase 15, without model dependencies."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

PHASE15_SEED = 20260826

Number = int | float
Label = TypeVar("Label", bound=object)


@dataclass(frozen=True)
class BootstrapResult:
    delta: float
    ci95: tuple[float, float]
    replicates: int
    seed: int
    confidence: float
    wins: int
    ties: int
    losses: int


@dataclass(frozen=True)
class RiskDifferenceResult:
    risk_difference: float
    ci95: tuple[float, float]
    discordant_pairs: dict[str, int]
    paired_bootstrap: BootstrapResult


def _paired_values(
    left: Sequence[Number], right: Sequence[Number]
) -> tuple[tuple[float, ...], ...]:
    if not left or not right:
        raise ValueError("paired inputs must be non-empty")
    if len(left) != len(right):
        raise ValueError("paired inputs must have equal length")
    return tuple(tuple(float(value) for value in pair) for pair in zip(left, right, strict=True))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def paired_bootstrap_delta(
    left: Sequence[Number],
    right: Sequence[Number],
    *,
    seed: int = PHASE15_SEED,
    replicates: int = 2000,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Return ``mean(left-right)`` and a deterministic paired bootstrap CI."""

    pairs = _paired_values(left, right)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    deltas = tuple(left_value - right_value for left_value, right_value in pairs)
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    samples = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(replicates)
    ]
    alpha = (1.0 - confidence) / 2.0
    ci = (_quantile(samples, alpha), _quantile(samples, 1.0 - alpha))
    wins = sum(delta > 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    losses = len(deltas) - wins - ties
    return BootstrapResult(
        delta=observed,
        ci95=ci,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
        wins=wins,
        ties=ties,
        losses=losses,
    )


def paired_risk_difference(
    before: Sequence[int],
    after: Sequence[int],
    *,
    seed: int = PHASE15_SEED,
    replicates: int = 2000,
    confidence: float = 0.95,
) -> RiskDifferenceResult:
    """Measure the paired binary risk change as ``after - before``."""

    pairs = _paired_values(before, after)
    if any(value not in {0.0, 1.0} for pair in pairs for value in pair):
        raise ValueError("paired risk inputs must be binary")
    bootstrap = paired_bootstrap_delta(
        after,
        before,
        seed=seed,
        replicates=replicates,
        confidence=confidence,
    )
    discordant = {
        "before_positive_after_negative": sum(a == 1.0 and b == 0.0 for a, b in pairs),
        "before_negative_after_positive": sum(a == 0.0 and b == 1.0 for a, b in pairs),
    }
    return RiskDifferenceResult(
        risk_difference=bootstrap.delta,
        ci95=bootstrap.ci95,
        discordant_pairs=discordant,
        paired_bootstrap=bootstrap,
    )


def paired_rank_biserial(left: Sequence[Number], right: Sequence[Number]) -> float:
    """Return the paired rank-biserial sign effect, in ``[-1, 1]``."""

    pairs = _paired_values(left, right)
    wins = sum(a > b for a, b in pairs)
    losses = sum(a < b for a, b in pairs)
    return (wins - losses) / len(pairs)


def cohens_kappa(left: Sequence[Label], right: Sequence[Label]) -> float:
    """Calculate unweighted Cohen's kappa for two paired categorical labels."""

    if not left or not right:
        raise ValueError("paired labels must be non-empty")
    if len(left) != len(right):
        raise ValueError("paired labels must have equal length")
    categories = set(left) | set(right)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    expected = sum(
        (sum(label == category for label in left) / len(left))
        * (sum(label == category for label in right) / len(right))
        for category in categories
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
