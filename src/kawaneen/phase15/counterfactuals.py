"""Offline citation-verifier and score-gate counterfactuals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .statistics import paired_risk_difference


def is_candidate_answer(record: Mapping[str, object]) -> bool:
    """Return whether a persisted record contains a schema-parsed answer decision."""

    result = record.get("result")
    return isinstance(result, Mapping) and str(result.get("decision", "")).lower() == "answer"


def citation_counterfactual(
    would_surface_without_verifier: Sequence[int],
    actual_verifier_outcome: Sequence[int],
    *,
    seed: int = 20260826,
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
        "relative_risk_reduction": ((before_rate - after_rate) / before_rate)
        if before_rate
        else None,
        "coverage_cost": before_rate - after_rate,
        "discordant_pairs": result.discordant_pairs,
        "seed": seed,
    }


def candidate_answer_counterfactual(
    candidate_answers: Sequence[int],
    verified_unsafe_answers: Sequence[int],
    *,
    defect_counts: Mapping[str, int] | None = None,
    seed: int = 20260826,
) -> dict[str, object]:
    """Measure verifier protection for candidate answers, never raw output presence.

    Both inputs are aligned candidate-answer rows.  A raw abstention or malformed
    response therefore cannot enter this population merely because it is non-empty.
    ``verified_unsafe_answers`` is sourced from persisted verifier/failure evidence.
    """

    result = citation_counterfactual(candidate_answers, verified_unsafe_answers, seed=seed)
    result.update(
        {
            "population_definition": "schema-parsed candidate answer decisions only",
            "defect_counts": dict(sorted((defect_counts or {}).items())),
        }
    )
    return result


def score_gate_sensitivity(
    scores: Sequence[float], *, outcomes: Mapping[str, Sequence[float]] | None = None
) -> dict[str, object]:
    if not scores:
        raise ValueError("score gate requires non-empty score distribution")
    ordered = sorted(float(value) for value in scores)
    quantiles = {"none": None, "bottom10": 0.10, "bottom25": 0.25, "bottom50": 0.50}
    result: dict[str, object] = {
        "method": "uncalibrated score-gate sensitivity analysis",
        "gates": {},
    }
    gates: dict[str, object] = {}
    for name, fraction in quantiles.items():
        threshold = (
            None
            if fraction is None
            else ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]
        )
        gate: dict[str, object] = {"threshold": threshold, "derived_without_relevance_labels": True}
        if outcomes:
            keep = (
                tuple(range(len(scores)))
                if fraction is None
                else tuple(index for index, score in enumerate(scores) if score > threshold)
            )
            gate["coverage"] = len(keep) / len(scores)
            gate["retained_count"] = len(keep)
            gate["quality"] = {
                key: (sum(values[index] for index in keep) / len(keep) if keep else None)
                for key, values in outcomes.items()
            }
        gates[name] = gate
    result["gates"] = gates
    return result
