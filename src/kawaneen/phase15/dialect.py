"""AI-generated dialect perturbation validation and paired DEV analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

from .contracts import Phase15Model
from .statistics import paired_bootstrap_delta


class DialectVariant(Phase15Model):
    variant_id: str
    base_intent_id: str
    dialect: str
    legal_intent_fingerprint: str
    qrel_fingerprint: str
    article_identifiers: tuple[str, ...] = ()
    date_identifiers: tuple[str, ...] = ()
    number_identifiers: tuple[str, ...] = ()
    text: str


def _numeric_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", value))


def validate_variants_before_outcomes(
    base_records: Mapping[str, Mapping[str, object]], variants: Sequence[DialectVariant]
) -> None:
    if len(variants) != 60 or len({item.variant_id for item in variants}) != 60:
        raise ValueError("exactly 60 unique dialect variants are required")
    counts = {dialect: sum(item.dialect == dialect for item in variants) for dialect in ("egyptian", "gulf_saudi", "levantine")}
    if counts != {"egyptian": 20, "gulf_saudi": 20, "levantine": 20}:
        raise ValueError("dialect variants require 20 variants per dialect")
    for item in variants:
        base = base_records.get(item.base_intent_id)
        if base is None:
            raise ValueError(f"variant references unknown base intent: {item.base_intent_id}")
        if item.legal_intent_fingerprint != str(base.get("legal_intent_fingerprint")):
            raise ValueError(f"legal intent changed for {item.variant_id}")
        if item.qrel_fingerprint != str(base.get("qrel_fingerprint")):
            raise ValueError(f"qrels changed for {item.variant_id}")
        if tuple(item.number_identifiers) != tuple(base.get("number_identifiers", ())):
            raise ValueError(f"numeric identifiers changed for {item.variant_id}")
        if tuple(item.date_identifiers) != tuple(base.get("date_identifiers", ())):
            raise ValueError(f"date identifiers changed for {item.variant_id}")
        if not item.text:
            raise ValueError(f"empty dialect text for {item.variant_id}")


def evaluate_dialect_runs(
    msa_by_system: Mapping[str, Mapping[str, Sequence[float]]],
    dialect_by_system: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    *,
    seed: int = 20260826,
) -> dict[str, object]:
    """Return dialect-minus-MSA paired deltas by dialect and retrieval system."""

    output: dict[str, object] = {"seed": seed, "dialects": {}}
    dialects: dict[str, object] = {}
    for dialect, system_runs in dialect_by_system.items():
        dialect_result: dict[str, object] = {}
        for system, metrics in system_runs.items():
            if system not in msa_by_system:
                raise ValueError(f"missing MSA control for system {system}")
            metric_result: dict[str, object] = {}
            for metric, values in metrics.items():
                baseline = msa_by_system[system].get(metric)
                if baseline is None:
                    raise ValueError(f"missing MSA metric {metric} for system {system}")
                metric_result[metric] = paired_bootstrap_delta(
                    values, baseline, seed=seed
                ).__dict__
            dialect_result[system] = metric_result
        dialects[dialect] = dialect_result
    output["dialects"] = dialects
    return output
