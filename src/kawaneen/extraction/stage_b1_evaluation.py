"""Offline evaluation of the frozen Phase 11B Stage-B1 DEV experiment."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from kawaneen.extraction.annotation import AnnotationRecord
from kawaneen.extraction.artifacts import write_private_json, write_text_free_json
from kawaneen.extraction.contracts import (
    ExactSourceSpan,
    ExtractionResult,
    Modality,
    NormativeRule,
)
from kawaneen.extraction.evaluation import score_spans
from kawaneen.extraction.hybrid import assemble_hybrid_result
from kawaneen.extraction.hybrid_runtime import _base_result
from kawaneen.extraction.orchestration import (
    HYBRID_STAGE_B1_CLEAN_CONFIG_PATH,
    HYBRID_STAGE_B1_CLEAN_RESULT_ROOT,
    PHASE11_CANDIDATE_COMPATIBLE_FINGERPRINT,
    PHASE11_CANDIDATE_POLICY_HASH,
    PHASE11_SELECTION_FINGERPRINT,
    PHASE11_SEMANTIC_RELEASE_FINGERPRINT,
    PHASE11_SEMANTIC_RELEASE_SHA256,
    _load_records,
)
from kawaneen.extraction.provider import parse_semantic_proposal

PRIVATE_EVALUATION_ROOT = Path(
    "artifacts/private/phase11_extraction/evaluation/hybrid-qwen-v1-stage-b1-clean/dev"
)
TRACKED_EVALUATION_PATH = Path("data/evaluation/phase11_hybrid_qwen_stage_b1_clean_dev_report.json")

SEMANTIC_FIELDS = (
    "regulated_entities",
    "actors",
    "actions",
    "conditions",
    "exceptions",
    "penalties",
    "deadlines",
    "effective_dates",
    "monetary_thresholds",
    "percentage_thresholds",
)
CANDIDATE_FIELDS = (
    "deadlines",
    "effective_dates",
    "monetary_thresholds",
    "percentage_thresholds",
)
MODALITIES = (Modality.OBLIGATION.value, Modality.PROHIBITION.value, Modality.PERMISSION.value)


def _span_key(span: ExactSourceSpan) -> tuple[str, int, int]:
    return span.text, span.start_char, span.end_char


def _spans(
    values: list[ExactSourceSpan] | tuple[ExactSourceSpan, ...],
) -> tuple[ExactSourceSpan, ...]:
    return tuple(values)


def _rule_spans(rules: tuple[NormativeRule, ...], field: str) -> tuple[ExactSourceSpan, ...]:
    values: list[ExactSourceSpan] = []
    for rule in rules:
        if field == "actors" and rule.actor_span is not None:
            values.append(rule.actor_span)
        elif field == "actions":
            values.append(rule.action_span)
        elif field == "conditions":
            values.extend(rule.condition_spans)
        elif field == "exceptions":
            values.extend(rule.exception_spans)
    return tuple(values)


def _field_spans(result: ExtractionResult, field: str) -> tuple[ExactSourceSpan, ...]:
    if field == "regulated_entities":
        return _spans(result.regulated_entities)
    if field == "penalties":
        return _spans(result.penalties)
    if field == "exceptions":
        return (*_spans(result.exceptions), *_rule_spans(result.rules, "exceptions"))
    if field in {"actors", "actions", "conditions"}:
        return _rule_spans(result.rules, field)
    raise ValueError(f"not a span field: {field}")


def _candidate_ids(result: ExtractionResult, field: str) -> tuple[str, ...]:
    return tuple(candidate.candidate_id for candidate in getattr(result, field))


def _candidate_metric(gold: tuple[str, ...], predicted: tuple[str, ...]) -> dict[str, float | int]:
    gold_set = set(gold)
    predicted_set = set(predicted)
    tp = len(gold_set & predicted_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "support": len(gold_set),
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def _empty_prediction(record: AnnotationRecord) -> ExtractionResult:
    return _base_result(record).model_copy(
        update={
            "configuration": "hybrid-qwen-v1",
            "extractor_version": "hybrid-qwen-v1",
            "rules": (),
            "obligations": (),
            "prohibitions": (),
            "permissions": (),
            "regulated_entities": (),
            "deadlines": (),
            "effective_dates": (),
            "monetary_thresholds": (),
            "percentage_thresholds": (),
            "exceptions": (),
            "penalties": (),
        }
    )


def _gold_result(record: AnnotationRecord) -> ExtractionResult:
    if record.human_annotations is None:
        raise ValueError(f"missing frozen annotation: {record.canonical_unit_id}")
    result = assemble_hybrid_result(
        record.canonical_text,
        _base_result(record),
        record.human_annotations.model_dump(mode="json"),
    )
    ExtractionResult.model_validate(result.model_dump(mode="json"))
    return result


def _result_path(record_id: str) -> Path:
    return HYBRID_STAGE_B1_CLEAN_RESULT_ROOT / (
        hashlib.sha256(record_id.encode("utf-8")).hexdigest() + ".json"
    )


def _load_prediction(record_id: str) -> tuple[ExtractionResult | None, dict[str, Any]]:
    path = _result_path(record_id)
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("lifecycle_state") != "complete":
        return None, payload
    result = ExtractionResult.model_validate(payload["result"])
    return result, payload


def _rule_key(rule: NormativeRule) -> tuple[Any, ...]:
    return (
        rule.modality.value,
        _span_key(rule.actor_span) if rule.actor_span else None,
        _span_key(rule.action_span),
        tuple(_span_key(item) for item in rule.condition_spans),
        tuple(_span_key(item) for item in rule.exception_spans),
        tuple(rule.deadline_refs),
        tuple(rule.monetary_threshold_refs),
        tuple(rule.percentage_threshold_refs),
    )


def _rule_exact_counts(
    gold: tuple[NormativeRule, ...], predicted: tuple[NormativeRule, ...]
) -> tuple[int, int, int]:
    gold_keys = Counter(_rule_key(item) for item in gold)
    predicted_keys = Counter(_rule_key(item) for item in predicted)
    tp = sum((gold_keys & predicted_keys).values())
    return tp, sum(predicted_keys.values()) - tp, sum(gold_keys.values()) - tp


def _rule_metrics(
    gold: tuple[NormativeRule, ...], predicted: tuple[NormativeRule, ...]
) -> dict[str, Any]:
    actor = score_spans(
        tuple(item.actor_span for item in gold if item.actor_span is not None),
        tuple(item.actor_span for item in predicted if item.actor_span is not None),
    )
    action = score_spans(
        tuple(item.action_span for item in gold), tuple(item.action_span for item in predicted)
    )
    comparable = min(len(gold), len(predicted))
    confusion: dict[str, dict[str, int]] = {
        modality: {other: 0 for other in MODALITIES} for modality in MODALITIES
    }
    for index in range(comparable):
        confusion[gold[index].modality.value][predicted[index].modality.value] += 1
    exact_tp, exact_fp, exact_fn = _rule_exact_counts(gold, predicted)
    exact_precision = exact_tp / (exact_tp + exact_fp) if exact_tp + exact_fp else 0.0
    exact_recall = exact_tp / (exact_tp + exact_fn) if exact_tp + exact_fn else 0.0
    exact_f1 = (
        2 * exact_precision * exact_recall / (exact_precision + exact_recall)
        if exact_precision + exact_recall
        else 0.0
    )
    gold_modality_support = Counter(item.modality.value for item in gold)
    predicted_modality_support = Counter(item.modality.value for item in predicted)
    return {
        "actor": actor,
        "action": action,
        "modality_accuracy": (
            sum(gold[index].modality is predicted[index].modality for index in range(comparable))
            / comparable
            if comparable
            else 0.0
        ),
        "modality_comparable": comparable,
        "modality_confusion": confusion,
        "gold_modality_support": dict(gold_modality_support),
        "predicted_modality_support": dict(predicted_modality_support),
        "full_rule_exact": {
            "TP": exact_tp,
            "FP": exact_fp,
            "FN": exact_fn,
            "support": len(gold),
            "precision": exact_precision,
            "recall": exact_recall,
            "F1": exact_f1,
        },
    }


def _merge_metrics(total: dict[str, int], current: Mapping[str, float | int]) -> None:
    for key in ("TP", "FP", "FN", "support"):
        total[key] += int(current[key])


def _finish_metric(total: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn = total["TP"], total["FP"], total["FN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        **total,
        "precision": precision,
        "recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _empty_semantic(result: ExtractionResult) -> bool:
    return not any(
        (
            result.regulated_entities,
            result.rules,
            result.deadlines,
            result.effective_dates,
            result.exceptions,
            result.penalties,
            result.monetary_thresholds,
            result.percentage_thresholds,
        )
    )


def _proposal_counts(payload: dict[str, Any]) -> tuple[int, int, int]:
    raw = payload.get("raw_provider_response")
    if not isinstance(raw, str):
        return 0, 0, 0
    try:
        proposal = parse_semantic_proposal(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0, 0, 0
    spans = len(proposal.regulated_entities) + len(proposal.exceptions) + len(proposal.penalties)
    refs = len(proposal.deadline_refs) + len(proposal.effective_date_refs)
    refs += len(proposal.monetary_threshold_refs) + len(proposal.percentage_threshold_refs)
    for rule in proposal.rules:
        spans += 1 + (rule.actor is not None) + len(rule.conditions) + len(rule.exceptions)
        refs += (
            len(rule.deadline_refs)
            + len(rule.effective_date_refs)
            + len(rule.monetary_threshold_refs)
            + len(rule.percentage_threshold_refs)
        )
    return spans, refs, len(proposal.rules)


def _diagnostics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return []
    result = cast(Mapping[str, Any], result)
    metadata = cast(Mapping[str, Any] | None, result.get("validation_metadata"))
    if not isinstance(metadata, Mapping):
        return []
    raw = cast(object, metadata.get("diagnostics"))
    if not isinstance(raw, list):
        return []
    raw_items = cast(list[object], raw)
    return [cast(dict[str, Any], item) for item in raw_items if isinstance(item, dict)]


def _diagnostic_counts(payload: dict[str, Any]) -> Counter[str]:
    diagnostics = _diagnostics(payload)
    counts: Counter[str] = Counter()
    for item in diagnostics:
        if isinstance(item.get("code"), str):
            counts[item["code"]] += 1
            field = str(item.get("field_name", ""))
            if field.startswith("regulated_entities"):
                counts["entities_dropped"] += 1
            elif ".conditions" in field:
                counts["conditions_dropped"] += 1
            elif ".exceptions" in field or field.startswith("exceptions"):
                counts["exceptions_dropped"] += 1
            elif field.startswith("penalties"):
                counts["penalties_dropped"] += 1
            if field.endswith(".action") and item["code"] in {
                "UNSUPPORTED_MODEL_SPAN",
                "AMBIGUOUS_OR_INVALID_OCCURRENCE",
            }:
                counts["invalid_action_rules_dropped"] += 1
    return counts


def _aggregate_view(
    pairs: list[tuple[AnnotationRecord, ExtractionResult, ExtractionResult | None]],
    *,
    denominator: int,
) -> dict[str, Any]:
    field_totals = {field: {"TP": 0, "FP": 0, "FN": 0, "support": 0} for field in SEMANTIC_FIELDS}
    candidate_totals = {
        field: {"TP": 0, "FP": 0, "FN": 0, "support": 0} for field in CANDIDATE_FIELDS
    }
    gold_rules = predicted_rules = 0
    actor_total = {"TP": 0, "FP": 0, "FN": 0, "support": 0}
    action_total = {"TP": 0, "FP": 0, "FN": 0, "support": 0}
    full_rule_total = {"TP": 0, "FP": 0, "FN": 0, "support": 0}
    modality_confusion = {gold: {pred: 0 for pred in MODALITIES} for gold in MODALITIES}
    modality_gold_support = {modality: 0 for modality in MODALITIES}
    modality_predicted_support = {modality: 0 for modality in MODALITIES}
    modality_comparable = 0
    modality_correct = 0
    clause_exact = 0
    per_record: list[dict[str, Any]] = []
    for record, gold, predicted in pairs:
        actual = predicted if predicted is not None else _empty_prediction(record)
        gold_rules += len(gold.rules)
        predicted_rules += len(actual.rules)
        field_record: dict[str, Any] = {}
        for field in SEMANTIC_FIELDS:
            if field in {
                "actors",
                "actions",
                "conditions",
                "exceptions",
                "regulated_entities",
                "penalties",
            }:
                current = score_spans(_field_spans(gold, field), _field_spans(actual, field))
            else:
                current = _candidate_metric(
                    _candidate_ids(gold, field), _candidate_ids(actual, field)
                )
            _merge_metrics(field_totals[field], current)
            field_record[field] = current
        for field in CANDIDATE_FIELDS:
            current = _candidate_metric(_candidate_ids(gold, field), _candidate_ids(actual, field))
            _merge_metrics(candidate_totals[field], current)
        rule = _rule_metrics(gold.rules, actual.rules)
        for key in ("TP", "FP", "FN", "support"):
            actor_total[key] += int(rule["actor"][key])
            action_total[key] += int(rule["action"][key])
            full_rule_total[key] += int(rule["full_rule_exact"][key])
        modality_comparable += int(rule["modality_comparable"])
        modality_correct += sum(
            rule["modality_confusion"][modality][modality] for modality in MODALITIES
        )
        for gold_modality in MODALITIES:
            for predicted_modality in MODALITIES:
                modality_confusion[gold_modality][predicted_modality] += rule["modality_confusion"][
                    gold_modality
                ][predicted_modality]
        for modality in MODALITIES:
            modality_gold_support[modality] += int(rule["gold_modality_support"].get(modality, 0))
            modality_predicted_support[modality] += int(
                rule["predicted_modality_support"].get(modality, 0)
            )
        exact_clause = predicted is not None and {_rule_key(item) for item in gold.rules} == {
            _rule_key(item) for item in actual.rules
        }
        clause_exact += exact_clause
        field_record.update(
            {
                "record_id": record.canonical_unit_id,
                "pipeline_complete": predicted is not None,
                "gold_rule_count": len(gold.rules),
                "predicted_rule_count": len(actual.rules),
                "clause_exact_match": exact_clause,
                "modality_accuracy": rule["modality_accuracy"],
            }
        )
        per_record.append(field_record)
    finished_fields = {field: _finish_metric(total) for field, total in field_totals.items()}
    candidate_metrics = {field: _finish_metric(total) for field, total in candidate_totals.items()}
    for metric in (*finished_fields.values(), *candidate_metrics.values()):
        if metric["support"] == 0:
            metric["recall_status"] = "not_estimable"
            metric["F1_status"] = "not_estimable"
    micro_total = {"TP": 0, "FP": 0, "FN": 0, "support": 0}
    for field in SEMANTIC_FIELDS:
        _merge_metrics(micro_total, field_totals[field])
    micro = _finish_metric(micro_total)
    macro_f1 = sum(float(metric["F1"]) for metric in finished_fields.values()) / len(
        finished_fields
    )
    full_rule = _finish_metric(full_rule_total)
    modality_by_class: dict[str, dict[str, float | int | str]] = {}
    for modality in MODALITIES:
        gold_support = modality_gold_support[modality]
        predicted_support = modality_predicted_support[modality]
        matched = modality_confusion[modality][modality]
        modality_metric = _finish_metric(
            {
                "TP": matched,
                "FP": predicted_support - matched,
                "FN": gold_support - matched,
                "support": gold_support,
            }
        )
        modality_metric["matched"] = matched
        modality_metric["recall_status"] = "not_estimable" if gold_support == 0 else "estimable"
        modality_by_class[modality] = modality_metric
    return {
        "record_count": denominator,
        "records_with_predictions": sum(predicted is not None for _, _, predicted in pairs),
        "gold_rule_count": gold_rules,
        "predicted_rule_count": predicted_rules,
        "field_metrics": finished_fields,
        "candidate_selection_metrics": candidate_metrics,
        "normative": {
            "actor": _finish_metric(actor_total),
            "action": _finish_metric(action_total),
            "modality_accuracy": modality_correct / modality_comparable
            if modality_comparable
            else 0.0,
            "modality_comparable": modality_comparable,
            "modality_confusion": modality_confusion,
            "modality_unmatched_gold": {
                modality: modality_gold_support[modality]
                - sum(modality_confusion[modality].values())
                for modality in MODALITIES
            },
            "modality_unmatched_predicted": {
                modality: modality_predicted_support[modality]
                - sum(modality_confusion[other][modality] for other in MODALITIES)
                for modality in MODALITIES
            },
            "modality_by_class": modality_by_class,
            "full_rule_exact_match": full_rule,
        },
        "micro": micro,
        "macro_F1": macro_f1,
        "clause_exact_match": {
            "matched_records": clause_exact,
            "rate": clause_exact / denominator if denominator else 0.0,
        },
        "per_record": per_record,
    }


def _complexity_breakdown(
    pairs: list[tuple[AnnotationRecord, ExtractionResult, ExtractionResult | None]],
) -> dict[str, Any]:
    buckets: dict[
        str, dict[str, list[tuple[AnnotationRecord, ExtractionResult, ExtractionResult | None]]]
    ] = {
        "source_length": defaultdict(list),
        "gold_rule_count": defaultdict(list),
        "candidate_count": defaultdict(list),
    }
    for item in pairs:
        record, gold, _ = item
        length = len(record.canonical_text)
        length_bucket = (
            "<=250"
            if length <= 250
            else "251-500"
            if length <= 500
            else "501-1000"
            if length <= 1000
            else "1001-1500"
        )
        rules_bucket = (
            "0"
            if len(gold.rules) == 0
            else "1"
            if len(gold.rules) == 1
            else "2"
            if len(gold.rules) == 2
            else "3+"
        )
        candidate_count = len(record.candidate_registry.candidates)
        candidate_bucket = (
            "0"
            if candidate_count == 0
            else "1-2"
            if candidate_count <= 2
            else "3-5"
            if candidate_count <= 5
            else "6+"
        )
        buckets["source_length"][length_bucket].append(item)
        buckets["gold_rule_count"][rules_bucket].append(item)
        buckets["candidate_count"][candidate_bucket].append(item)
    output: dict[str, Any] = {}
    for dimension, groups in buckets.items():
        output[dimension] = {}
        for bucket, group in sorted(groups.items()):
            evaluated = _aggregate_view(group, denominator=len(group))
            output[dimension][bucket] = {
                "records": len(group),
                "completed_predictions": evaluated["records_with_predictions"],
                "semantic_micro_F1": evaluated["micro"]["F1"],
                "clause_exact_match_rate": evaluated["clause_exact_match"]["rate"],
            }
    return output


def evaluate_clean_stage_b1_dev() -> dict[str, Any]:
    """Evaluate only the locked DEV set; never loads HOLDOUT records."""

    records = _load_records("dev")
    if len(records) != 80 or any(record.split != "dev" for record in records):
        raise ValueError("Stage B1 evaluation requires exactly 80 DEV records")
    pairs: list[tuple[AnnotationRecord, ExtractionResult, ExtractionResult | None]] = []
    field_local = Counter[str]()
    proposal_span_count = proposal_ref_count = 0
    failed_record_ids: list[str] = []
    pipeline_failure_categories: Counter[str] = Counter()
    completed = 0
    valid_raw_schema = 0
    valid_final_schema = 0
    provenance_complete = 0
    valid_empty = 0
    pipeline_diagnostics: Counter[str] = Counter()
    raw_result_bytes: list[tuple[str, bytes]] = []
    for record in records:
        gold = _gold_result(record)
        predicted, payload = _load_prediction(record.canonical_unit_id)
        raw_result_bytes.append(
            (record.canonical_unit_id, _result_path(record.canonical_unit_id).read_bytes())
        )
        if predicted is None:
            failed_record_ids.append(record.canonical_unit_id)
            diagnostics = _diagnostics(payload)
            category = "other"
            for item in diagnostics:
                if item.get("code") == "INVALID_PROVIDER_JSON":
                    category = "INVALID_JSON_OUTPUT_TRUNCATED"
            pipeline_failure_categories[category] += 1
            pipeline_diagnostics.update(str(item.get("code", "UNKNOWN")) for item in diagnostics)
            pairs.append((record, gold, None))
            continue
        completed += 1
        if predicted.validation_metadata.raw_provider_schema_valid:
            valid_raw_schema += 1
        valid_final_schema += 1
        provenance_complete += bool(
            {item.field_name for item in predicted.field_provenance}
            >= {
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
            }
        )
        valid_empty += _empty_semantic(predicted)
        diagnostics = _diagnostic_counts(payload)
        field_local.update(diagnostics)
        spans, refs, _ = _proposal_counts(payload)
        proposal_span_count += spans
        proposal_ref_count += refs
        pairs.append((record, gold, predicted))
    # The failed record has no completed result; its raw response is counted once
    # as a provider/schema failure, never as a valid empty prediction.
    conditional_pairs = [item for item in pairs if item[2] is not None]
    conditional = _aggregate_view(conditional_pairs, denominator=completed)
    end_to_end = _aggregate_view(pairs, denominator=len(records))
    all_diagnostics = field_local + pipeline_diagnostics
    unsupported = all_diagnostics["UNSUPPORTED_MODEL_SPAN"]
    invalid_refs = all_diagnostics["INVALID_CANDIDATE_REFERENCE"]
    corrections = all_diagnostics["INVALID_OCCURRENCE_CORRECTED"]
    ambiguous = all_diagnostics["AMBIGUOUS_OR_INVALID_OCCURRENCE"]
    safety = {
        "PipelineCompletionRate": {"count": completed, "denominator": 80, "rate": completed / 80},
        "RawProviderSchemaValidityRate": {
            "count": valid_raw_schema,
            "denominator": 80,
            "rate": valid_raw_schema / 80,
        },
        "FinalSchemaValidityRate": {
            "completed_outputs": {
                "count": valid_final_schema,
                "denominator": completed,
                "rate": 1.0,
            },
            "end_to_end": {
                "count": valid_final_schema,
                "denominator": 80,
                "rate": valid_final_schema / 80,
            },
        },
        "UnsupportedSpanProposalRate": {
            "count": unsupported,
            "denominator": proposal_span_count,
            "rate": unsupported / proposal_span_count if proposal_span_count else 0.0,
        },
        "UnsupportedSpanAcceptanceRate": {"count": 0, "denominator": unsupported, "rate": 0.0},
        "InvalidCandidateReferenceProposalRate": {
            "count": invalid_refs,
            "denominator": proposal_ref_count,
            "rate": invalid_refs / proposal_ref_count if proposal_ref_count else 0.0,
        },
        "InvalidCandidateReferenceAcceptanceRate": {
            "count": 0,
            "denominator": invalid_refs,
            "rate": 0.0,
        },
        "ProvenanceCompletenessRate": {
            "completed_outputs": {
                "count": provenance_complete,
                "denominator": completed,
                "rate": provenance_complete / completed,
            },
            "end_to_end": {
                "count": provenance_complete,
                "denominator": 80,
                "rate": provenance_complete / 80,
            },
        },
    }
    diagnostics_report = {
        "invalid_candidate_references_proposed": invalid_refs,
        "invalid_candidate_references_dropped": invalid_refs,
        "unsupported_spans_proposed": unsupported,
        "unsupported_spans_dropped": unsupported,
        "unique_occurrence_server_corrections": corrections,
        "ambiguous_or_repeated_occurrences_rejected": ambiguous,
        "invalid_rules_dropped_because_action_invalid": field_local["invalid_action_rules_dropped"],
        "entities_dropped": field_local["entities_dropped"],
        "conditions_dropped": field_local["conditions_dropped"],
        "exceptions_dropped": field_local["exceptions_dropped"],
        "penalties_dropped": field_local["penalties_dropped"],
        "valid_empty_extraction_results": valid_empty,
        "diagnostic_code_counts": dict(sorted(all_diagnostics.items())),
    }
    taxonomy: Counter[str] = Counter()
    wrong_modality = 0
    association_mismatches = 0
    for _record, gold, predicted in pairs:
        if predicted is None:
            taxonomy["PIPELINE_FAILURE"] += 1
            continue
        if len(gold.rules) > len(predicted.rules):
            taxonomy["MISSED_EXTRACTION"] += len(gold.rules) - len(predicted.rules)
        if len(predicted.rules) > len(gold.rules):
            taxonomy["SPURIOUS_EXTRACTION"] += len(predicted.rules) - len(gold.rules)
        if any(
            set(_candidate_ids(gold, field)) != set(_candidate_ids(predicted, field))
            for field in CANDIDATE_FIELDS
        ):
            taxonomy["WRONG_CANDIDATE_CLASSIFICATION"] += 1
        if any(
            len(values) != len({_span_key(value) for value in values})
            for values in (
                _field_spans(predicted, "regulated_entities"),
                _field_spans(predicted, "actions"),
                _field_spans(predicted, "exceptions"),
                _field_spans(predicted, "penalties"),
            )
        ):
            taxonomy["DUPLICATE_EXTRACTION"] += 1
        comparable = min(len(gold.rules), len(predicted.rules))
        wrong_modality += sum(
            gold.rules[index].modality is not predicted.rules[index].modality
            for index in range(comparable)
        )
        predicted_by_action = {_span_key(rule.action_span): rule for rule in predicted.rules}
        for gold_rule in gold.rules:
            predicted_rule = predicted_by_action.get(_span_key(gold_rule.action_span))
            if predicted_rule is not None and (
                (_span_key(gold_rule.actor_span) if gold_rule.actor_span else None)
                != (_span_key(predicted_rule.actor_span) if predicted_rule.actor_span else None)
            ):
                association_mismatches += 1
    # Span-boundary and association categories are reported from strict field deltas.
    for field in ("actors", "actions", "conditions", "exceptions", "penalties"):
        metric = conditional["field_metrics"][field]
        if metric["TP"] and metric["FP"] and metric["FN"]:
            taxonomy["SPAN_BOUNDARY_ERROR"] += 1
    taxonomy.update({key: value for key, value in pipeline_failure_categories.items()})
    taxonomy["WRONG_MODALITY"] = wrong_modality
    taxonomy["WRONG_ACTOR_ACTION_ASSOCIATION"] = association_mismatches
    taxonomy["UNSUPPORTED_MODEL_SPAN"] = unsupported
    result_set_hash = hashlib.sha256(
        b"".join(
            record_id.encode("utf-8") + payload for record_id, payload in sorted(raw_result_bytes)
        )
    ).hexdigest()
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "phase11_hybrid_stage_b1_clean_dev_evaluation",
        "extractor": "hybrid-qwen-v1-stage-b1-clean",
        "split": "dev",
        "reference_status": "AI-reviewed/adjudicated reference; not human gold",
        "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
        "candidate_policy": "phase11-candidates-v3",
        "candidate_policy_hash": PHASE11_CANDIDATE_POLICY_HASH,
        "candidate_compatible_release_fingerprint": PHASE11_CANDIDATE_COMPATIBLE_FINGERPRINT,
        "template_sha256": "6cf49ae83d9917621d8487f91c0f176dfb316f2d0b3cf0f8867330ba6c1daf58",
        "schema_hash": "7e1e0287c0a384d09fddc419964e3422b95a1540d2a7fc92aca58fa692d03ee0",
        "annotation_release_sha256": PHASE11_SEMANTIC_RELEASE_SHA256,
        "annotation_release_fingerprint": PHASE11_SEMANTIC_RELEASE_FINGERPRINT,
        "record_hashes": {
            "clean_inference_result_set_sha256": result_set_hash,
            "clean_config_sha256": hashlib.sha256(
                HYBRID_STAGE_B1_CLEAN_CONFIG_PATH.read_bytes()
            ).hexdigest(),
        },
        "failure": {
            "record_id": failed_record_ids,
            "count": len(failed_record_ids),
            "category": dict(pipeline_failure_categories),
            "attempt_2_diagnostic_codes": dict(sorted(pipeline_diagnostics.items())),
        },
        "conditional_79_record_view": {
            key: value for key, value in conditional.items() if key != "per_record"
        },
        "end_to_end_80_record_view": {
            key: value for key, value in end_to_end.items() if key != "per_record"
        },
        "safety_structural_metrics": safety,
        "field_local_rejection_diagnostics": diagnostics_report,
        "error_taxonomy": dict(sorted(taxonomy.items())),
        "complexity_breakdown": _complexity_breakdown(pairs),
        "disposition": "HYBRID_STAGE_B1_EXPERIMENTAL",
        "historical_deterministic_semantic_zero": "superseded_diagnostic",
        "method_notes": {
            "micro_macro_fields": list(SEMANTIC_FIELDS),
            "candidate_fields_compare_semantic_candidate_ids": True,
            "failed_record_end_to_end_prediction": "empty_prediction_for_metrics_only",
            "rule_matching": (
                "strict exact rule-key matching; modality uses existing "
                "index-aligned evaluator behavior"
            ),
            "effective_date_rule_refs": (
                "not representable in current ExtractionResult NormativeRule; "
                "top-level effective_dates evaluated"
            ),
        },
    }
    private_payload = {
        **report,
        "per_record_conditional_and_end_to_end": {
            "conditional": conditional["per_record"],
            "end_to_end": end_to_end["per_record"],
        },
    }
    PRIVATE_EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    write_private_json(PRIVATE_EVALUATION_ROOT / "per_record_metrics.json", private_payload)
    write_private_json(
        PRIVATE_EVALUATION_ROOT / "failed_record_attempt_2_diagnostic.json",
        {
            "record_id": failed_record_ids[0] if failed_record_ids else None,
            "category": pipeline_failure_categories,
            "checkpoint": json.loads(
                (
                    Path(
                        "artifacts/private/phase11_extraction/checkpoints/hybrid-qwen-v1-stage-b1-clean/dev"
                    )
                    / f"{failed_record_ids[0]}.json"
                ).read_text(encoding="utf-8")
            )
            if failed_record_ids
            else None,
        },
    )
    tracked_report = json.loads(json.dumps(report))
    write_text_free_json(TRACKED_EVALUATION_PATH, tracked_report)
    report["tracked_evaluation_sha256"] = hashlib.sha256(
        TRACKED_EVALUATION_PATH.read_bytes()
    ).hexdigest()
    write_private_json(PRIVATE_EVALUATION_ROOT / "evaluation_report_private.json", report)
    return report


__all__ = ["evaluate_clean_stage_b1_dev"]
