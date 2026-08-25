"""Offline citation-verifier and score-gate counterfactuals."""

from __future__ import annotations

from typing import Mapping, Sequence

from .statistics import paired_risk_difference


def citation_counterfactual(
    would_surface_without_verifier: Sequence[int], actual_verifier_outcome: Sequence[int], *, seed: int = 20260826
) -> dict[str, object]:
    if len(would_surface_without_verifier) != len(actual_verifier_outcome):
        raise ValueError("counterfactual pairs must be aligned")
    result = paired_risk_difference(
        would_surface_without_verifier, actual_verifier_outcome, seed=seed
    )
    before_rate = sum(would_surface_without_verifier) / len(would_surface_without_verifier)
    after_rate = sum(actual_verifier_outcome) / len(actual_verifier_outcome)
    before_rate = float(before_rate)
    return {
        "pre_unsafe_acceptance": before_rate,
        "post_unsafe_acceptance": after_rate,
        "absolute_risk_reduction": -result.risk_difference,
        "absolute_risk_reduction_ci95": (-result.ci95[1], -result.ci95[0]),
        "relative_risk_reduction": ((before_rate - after_rate) / before_rate) if before_rate else None,
        "coverage_cost": before_rate - after_rate,
        "discordant_pairs": result.discordant_pairs,
        "seed": seed,
    }


def score_gate_sensitivity(scores: Sequence[float], *, outcomes: Mapping[str, Sequence[float]] | None = None) -> dict[str, object]:
    if not scores:
        raise ValueError("score gate requires non-empty score distribution")
    ordered = sorted(float(value) for value in scores)
    quantiles = {"none": None, "bottom10": 0.10, "bottom25": 0.25, "bottom50": 0.50}
    result: dict[str, object] = {"method": "uncalibrated score-gate sensitivity analysis", "gates": {}}
    gates: dict[str, object] = {}
    for name, fraction in quantiles.items():
        threshold = None if fraction is None else ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]
        gate: dict[str, object] = {"threshold": threshold, "derived_without_relevance_labels": True}
        if outcomes:
            gate["quality"] = {
                key: sum(values) / len(values) if values else None for key, values in outcomes.items()
            }
        gates[name] = gate
    result["gates"] = gates
    return result
