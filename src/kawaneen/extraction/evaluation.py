"""Offline strict-span evaluation for Phase 11A."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from kawaneen.extraction.contracts import (
    Candidate,
    ExactSourceSpan,
    ExtractionResult,
    NormativeRule,
)

EVALUATED_FIELDS = (
    "regulated_entities",
    "deadlines",
    "effective_dates",
    "penalties",
    "monetary_thresholds",
    "percentage_thresholds",
    "exceptions",
    "referenced_articles",
    "referenced_regulations",
)
REQUIRED_PROVENANCE_FIELDS = (
    "issuing_authority",
    "regulated_entities",
    "rules",
    "deadlines",
    "effective_dates",
    "penalties",
    "monetary_thresholds",
    "percentage_thresholds",
    "exceptions",
    "referenced_articles",
    "referenced_regulations",
)


def _key(span: ExactSourceSpan) -> tuple[str, int, int]:
    return span.text, span.start_char, span.end_char


def _as_spans(values: Iterable[object]) -> tuple[ExactSourceSpan, ...]:
    spans: list[ExactSourceSpan] = []
    for value in values:
        if isinstance(value, ExactSourceSpan):
            spans.append(value)
        elif isinstance(value, Candidate):
            spans.append(value.span)
    return tuple(spans)


def score_spans(
    gold: Iterable[ExactSourceSpan],
    predicted: Iterable[ExactSourceSpan],
) -> dict[str, float | int]:
    gold_keys = {_key(item) for item in gold}
    predicted_keys = {_key(item) for item in predicted}
    tp = len(gold_keys & predicted_keys)
    fp = len(predicted_keys - gold_keys)
    fn = len(gold_keys - predicted_keys)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "support": len(gold_keys),
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def _rule_key(rule: NormativeRule) -> tuple[object, ...]:
    return (
        rule.modality.value,
        _key(rule.actor_span) if rule.actor_span else None,
        _key(rule.action_span),
        tuple(_key(item) for item in rule.condition_spans),
        tuple(_key(item) for item in rule.exception_spans),
        rule.deadline_refs,
        rule.monetary_threshold_refs,
        rule.percentage_threshold_refs,
    )


def _rule_metric(
    gold: tuple[NormativeRule, ...], predicted: tuple[NormativeRule, ...]
) -> dict[str, float]:
    exact = len({_rule_key(item) for item in gold} & {_rule_key(item) for item in predicted})
    full = exact / max(len(gold), len(predicted), 1)
    actor = score_spans(
        tuple(item.actor_span for item in gold if item.actor_span is not None),
        tuple(item.actor_span for item in predicted if item.actor_span is not None),
    )
    action = score_spans(
        tuple(item.action_span for item in gold), tuple(item.action_span for item in predicted)
    )
    comparable = min(len(gold), len(predicted))
    modality_accuracy = (
        sum(gold[index].modality is predicted[index].modality for index in range(comparable))
        / comparable
        if comparable
        else 0.0
    )
    return {
        "actor_span_F1": float(actor["F1"]),
        "action_span_F1": float(action["F1"]),
        "modality_accuracy": modality_accuracy,
        "full_rule_exact_match": full,
    }


def _provenance_complete(result: ExtractionResult) -> bool:
    names = {item.field_name for item in result.field_provenance}
    return set(REQUIRED_PROVENANCE_FIELDS).issubset(names)


def evaluate_extractions(gold: ExtractionResult, predicted: ExtractionResult) -> dict[str, Any]:
    fields: dict[str, dict[str, float | int]] = {}
    for field_name in EVALUATED_FIELDS:
        fields[field_name] = score_spans(
            _as_spans(getattr(gold, field_name)),
            _as_spans(getattr(predicted, field_name)),
        )
    totals = {
        name: sum(int(fields[field][name]) for field in fields)
        for name in ("TP", "FP", "FN", "support")
    }
    precision = totals["TP"] / (totals["TP"] + totals["FP"]) if totals["TP"] + totals["FP"] else 0.0
    recall = totals["TP"] / (totals["TP"] + totals["FN"]) if totals["TP"] + totals["FN"] else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro_f1 = sum(float(value["F1"]) for value in fields.values()) / len(fields)
    gold_rules = gold.rules
    predicted_rules = predicted.rules
    error_counts: Counter[str] = Counter()
    for value in fields.values():
        if value["FN"] and value["FP"]:
            error_counts["SPAN_BOUNDARY_ERROR"] += 1
        elif value["FN"]:
            error_counts["MISSED_EXTRACTION"] += int(value["FN"])
        elif value["FP"]:
            error_counts["SPURIOUS_EXTRACTION"] += int(value["FP"])
    return {
        "field_metrics": fields,
        "micro": {"precision": precision, "recall": recall, "F1": micro_f1},
        "aggregate": {"macro_F1": macro_f1},
        "clause_exact_match_accuracy": float(
            _rule_key(gold_rules[0]) == _rule_key(predicted_rules[0])
        )
        if len(gold_rules) == len(predicted_rules) == 1
        else float(
            {_rule_key(item) for item in gold_rules}
            == {_rule_key(item) for item in predicted_rules}
        ),
        "rule_metrics": _rule_metric(gold_rules, predicted_rules),
        "error_counts": dict(sorted(error_counts.items())),
        "engineering_metrics": {
            "RawProviderSchemaValidityRate": 1.0
            if predicted.validation_metadata.raw_provider_schema_valid
            else 0.0,
            "FinalSchemaValidityRate": 1.0,
            "UnsupportedSpanProposalRate": sum(
                item.code == "UNSUPPORTED_MODEL_SPAN"
                for item in predicted.validation_metadata.diagnostics
            )
            / max(len(predicted.validation_metadata.diagnostics), 1),
            "UnsupportedSpanAcceptanceRate": 0.0,
            "InvalidCandidateReferenceRate": sum(
                item.code == "INVALID_CANDIDATE_REFERENCE"
                for item in predicted.validation_metadata.diagnostics
            )
            / max(len(predicted.validation_metadata.diagnostics), 1),
            "InvalidCandidateReferenceAcceptanceRate": 0.0,
            "ProvenanceCompletenessRate": 1.0 if _provenance_complete(predicted) else 0.0,
        },
    }
