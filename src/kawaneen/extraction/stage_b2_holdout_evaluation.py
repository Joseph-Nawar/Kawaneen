"""Offline evaluation of the frozen Phase 11B Stage-B2 HOLDOUT run."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from kawaneen.extraction.annotation import AnnotationRecord
from kawaneen.extraction.artifacts import write_private_json, write_text_free_json
from kawaneen.extraction.contracts import ExtractionResult
from kawaneen.extraction.hybrid_prompt import HYBRID_STAGE_B2_PROMPT_TEMPLATE_VERSION
from kawaneen.extraction.orchestration import (
    HOLDOUT_RELEASE_MANIFEST_PATH,
    HOLDOUT_RELEASE_PATH,
    HYBRID_STAGE_B2_HOLDOUT_CONFIG_PATH,
    HYBRID_STAGE_B2_HOLDOUT_RESULT_ROOT,
    PHASE11_CANDIDATE_POLICY_HASH,
    PHASE11_SELECTION_FINGERPRINT,
    _load_records,
)
from kawaneen.extraction.provider import parse_semantic_proposal
from kawaneen.extraction.stage_b1_evaluation import (
    _aggregate_view,
    _complexity_breakdown,
    _diagnostic_counts,
    _diagnostics,
    _empty_semantic,
    _gold_result,
    _proposal_counts,
)

PRIVATE_EVALUATION_ROOT = Path(
    "artifacts/private/phase11_extraction/evaluation/hybrid-qwen-v1-stage-b2-clean/holdout"
)
TRACKED_EVALUATION_PATH = Path(
    "data/evaluation/phase11_hybrid_qwen_stage_b2_clean_holdout_report.json"
)
HOLDOUT_CHECKPOINT_ROOT = Path(
    "artifacts/private/phase11_extraction/checkpoints/hybrid-qwen-v1-stage-b2-clean/holdout"
)
DEV_REPORT_PATH = Path("data/evaluation/phase11_hybrid_qwen_stage_b2_clean_dev_report.json")
SELECTED_CONFIG_PATH = Path("data/manifests/extraction/phase11_selected_configuration_v1.json")
SCHEMA_HASH = "7e1e0287c0a384d09fddc419964e3422b95a1540d2a7fc92aca58fa692d03ee0"
PROMPT_HASH = "72d8e6b613b56ddb61cf536df55760b2220f6e570c2f09b8536e7263d0c1f9bf"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
MODEL_DIGEST = "sha256:0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
MODALITIES = ("obligation", "prohibition", "permission")


def _result_path(record_id: str) -> Path:
    return HYBRID_STAGE_B2_HOLDOUT_RESULT_ROOT / (
        hashlib.sha256(record_id.encode("utf-8")).hexdigest() + ".json"
    )


def _load_prediction(record_id: str) -> tuple[ExtractionResult | None, dict[str, Any]]:
    path = _result_path(record_id)
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("lifecycle_state") != "complete":
        return None, payload
    return ExtractionResult.model_validate(payload["result"]), payload


def _valid_provenance(result: ExtractionResult) -> bool:
    return {item.field_name for item in result.field_provenance} >= {
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


def _raw_proposal(payload: dict[str, Any]) -> Any | None:
    raw = payload.get("raw_provider_response")
    if not isinstance(raw, str):
        return None
    try:
        return parse_semantic_proposal(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _provider_body_empty(proposal: Any) -> bool:
    return not any(
        (
            proposal.regulated_entities,
            proposal.rules,
            proposal.exceptions,
            proposal.penalties,
            proposal.deadline_refs,
            proposal.effective_date_refs,
            proposal.monetary_threshold_refs,
            proposal.percentage_threshold_refs,
        )
    )


def _failure_diagnostic(record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("raw_provider_response")
    result = cast(dict[str, Any], payload.get("result", {}))
    metadata = cast(dict[str, Any], result.get("validation_metadata", {}))
    checkpoint_path = HOLDOUT_CHECKPOINT_ROOT / f"{record_id}.json"
    checkpoint = cast(dict[str, Any], json.loads(checkpoint_path.read_text(encoding="utf-8")))
    failure_reason = str(checkpoint.get("failure_reason") or payload.get("failure") or "")
    raw_text = raw if isinstance(raw, str) else ""
    parse_status = "valid"
    try:
        json.loads(raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        parse_status = "invalid"
    truncated = any(
        marker in str(metadata.get("diagnostics", [{}])[0].get("message", ""))
        for marker in ("Unterminated", "Expecting ',' delimiter", "Expecting value")
        if isinstance(metadata.get("diagnostics"), list) and metadata["diagnostics"]
    )
    return {
        "record_id": record_id,
        "raw_response_received": isinstance(raw, str),
        "raw_response_length": len(raw_text),
        "json_parse_status": parse_status,
        "provider_schema_status": (
            "valid" if metadata.get("raw_provider_schema_valid") else "invalid"
        ),
        "truncation_evidence": truncated,
        "exact_span_stage": "not_reached",
        "candidate_reference_stage": "not_reached",
        "final_failure_category": "INVALID_PROVIDER_JSON",
        "failure_subcategory": "OUTPUT_TRUNCATED" if truncated else "INVALID_JSON",
        "failure_reason": failure_reason,
        "attempt_count": checkpoint.get("attempt_count"),
        "attempt_history": checkpoint.get("attempt_history", []),
        "prior_failure_type": checkpoint.get("prior_failure_type"),
        "retry_reason": checkpoint.get("retry_reason"),
    }


def _taxonomy(
    pairs: list[tuple[AnnotationRecord, ExtractionResult, ExtractionResult | None]],
    diagnostics: Counter[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _record, gold, predicted in pairs:
        if predicted is None:
            counts["PIPELINE_FAILURE"] += 1
            continue
        if len(gold.rules) > len(predicted.rules):
            counts["MISSED_EXTRACTION"] += len(gold.rules) - len(predicted.rules)
        if len(predicted.rules) > len(gold.rules):
            counts["SPURIOUS_EXTRACTION"] += len(predicted.rules) - len(gold.rules)
        if any(
            set(candidate.candidate_id for candidate in getattr(gold, field))
            != set(candidate.candidate_id for candidate in getattr(predicted, field))
            for field in (
                "deadlines",
                "effective_dates",
                "monetary_thresholds",
                "percentage_thresholds",
            )
        ):
            counts["WRONG_CANDIDATE_CLASSIFICATION"] += 1
        for field_values in (
            predicted.regulated_entities,
            predicted.exceptions,
            predicted.penalties,
        ):
            if len(field_values) != len({_span_key(item) for item in field_values}):
                counts["DUPLICATE_EXTRACTION"] += 1
        comparable = min(len(gold.rules), len(predicted.rules))
        counts["WRONG_MODALITY"] += sum(
            gold.rules[index].modality is not predicted.rules[index].modality
            for index in range(comparable)
        )
        predicted_by_action = {_span_key(rule.action_span): rule for rule in predicted.rules}
        counts["WRONG_ACTOR_ACTION_ASSOCIATION"] += sum(
            1
            for gold_rule in gold.rules
            if (predicted_rule := predicted_by_action.get(_span_key(gold_rule.action_span)))
            is not None
            and (_span_key(gold_rule.actor_span) if gold_rule.actor_span else None)
            != (_span_key(predicted_rule.actor_span) if predicted_rule.actor_span else None)
        )
    for field in ("actors", "actions", "conditions", "exceptions", "penalties"):
        if diagnostics[field] > 0:
            counts["SPAN_BOUNDARY_ERROR"] += 1 if diagnostics[field] else 0
    counts["UNSUPPORTED_MODEL_SPAN"] = diagnostics["UNSUPPORTED_MODEL_SPAN"]
    for category in (
        "MISSED_EXTRACTION",
        "SPURIOUS_EXTRACTION",
        "SPAN_BOUNDARY_ERROR",
        "WRONG_MODALITY",
        "WRONG_ACTOR_ACTION_ASSOCIATION",
        "WRONG_CANDIDATE_CLASSIFICATION",
        "DUPLICATE_EXTRACTION",
        "UNSUPPORTED_MODEL_SPAN",
        "PIPELINE_FAILURE",
        "ANNOTATION_AMBIGUITY",
    ):
        counts.setdefault(category, 0)
    return dict(sorted(counts.items()))


def _assert_frozen_inputs() -> dict[str, Any]:
    manifest = cast(dict[str, Any], json.loads(HOLDOUT_RELEASE_MANIFEST_PATH.read_text()))
    private_sha = hashlib.sha256(HOLDOUT_RELEASE_PATH.read_bytes()).hexdigest()
    if manifest.get("status") != "HOLDOUT_REFERENCE_FROZEN" or manifest.get("record_count") != 40:
        raise ValueError("HOLDOUT reference is not the frozen 40-record release")
    if manifest.get("private_release_sha256") != private_sha:
        raise ValueError("HOLDOUT private release hash does not match its manifest")
    if manifest.get("selection_fingerprint") != PHASE11_SELECTION_FINGERPRINT:
        raise ValueError("HOLDOUT selection fingerprint is not locked")
    return manifest


def _span_key(span: Any) -> tuple[str, int, int]:
    return span.text, span.start_char, span.end_char


def evaluate_clean_stage_b2_holdout() -> dict[str, Any]:
    """Evaluate the frozen HOLDOUT outputs without invoking a provider."""

    manifest = _assert_frozen_inputs()
    records = _load_records("holdout")
    if len(records) != 40:
        raise ValueError("HOLDOUT evaluation requires exactly 40 records")
    pairs: list[tuple[AnnotationRecord, ExtractionResult, ExtractionResult | None]] = []
    field_local: Counter[str] = Counter()
    pipeline_diagnostics: Counter[str] = Counter()
    proposal_span_count = proposal_ref_count = 0
    completed = valid_raw_schema = valid_final_schema = provenance_complete = 0
    valid_empty = provider_bodies_empty = raw_entities = raw_rules = 0
    final_entities = final_rules = 0
    failed_diagnostics: list[dict[str, Any]] = []
    raw_result_bytes: list[tuple[str, bytes]] = []
    for record in records:
        gold = _gold_result(record)
        predicted, payload = _load_prediction(record.canonical_unit_id)
        raw_result_bytes.append(
            (record.canonical_unit_id, _result_path(record.canonical_unit_id).read_bytes())
        )
        proposal = _raw_proposal(payload)
        if proposal is not None:
            provider_bodies_empty += int(_provider_body_empty(proposal))
            raw_entities += len(proposal.regulated_entities)
            raw_rules += len(proposal.rules)
        spans, refs, _ = _proposal_counts(payload)
        proposal_span_count += spans
        proposal_ref_count += refs
        result_payload = cast(dict[str, Any], payload.get("result", {}))
        metadata = cast(dict[str, Any], result_payload.get("validation_metadata", {}))
        valid_raw_schema += int(bool(metadata.get("raw_provider_schema_valid")))
        if predicted is None:
            failed_diagnostics.append(_failure_diagnostic(record.canonical_unit_id, payload))
            pipeline_diagnostics.update(
                str(item.get("code", "UNKNOWN")) for item in _diagnostics(payload)
            )
            pairs.append((record, gold, None))
            continue
        completed += 1
        valid_final_schema += 1
        provenance_complete += int(_valid_provenance(predicted))
        valid_empty += int(_empty_semantic(predicted))
        final_entities += len(predicted.regulated_entities)
        final_rules += len(predicted.rules)
        diagnostics = _diagnostic_counts(payload)
        field_local.update(diagnostics)
        pairs.append((record, gold, predicted))

    conditional_pairs = [item for item in pairs if item[2] is not None]
    conditional = _aggregate_view(conditional_pairs, denominator=completed)
    end_to_end = _aggregate_view(pairs, denominator=len(records))
    all_diagnostics = field_local + pipeline_diagnostics
    unsupported = all_diagnostics["UNSUPPORTED_MODEL_SPAN"]
    invalid_refs = all_diagnostics["INVALID_CANDIDATE_REFERENCE"]
    corrections = all_diagnostics["INVALID_OCCURRENCE_CORRECTED"]
    ambiguous = all_diagnostics["AMBIGUOUS_OR_INVALID_OCCURRENCE"]
    safety = {
        "PipelineCompletionRate": {"count": completed, "denominator": 40, "rate": completed / 40},
        "RawProviderSchemaValidityRate": {
            "count": valid_raw_schema,
            "denominator": 40,
            "rate": valid_raw_schema / 40,
        },
        "FinalSchemaValidityRate": {
            "completed_outputs": {
                "count": valid_final_schema,
                "denominator": completed,
                "rate": 1.0,
            },
            "end_to_end": {
                "count": valid_final_schema,
                "denominator": 40,
                "rate": valid_final_schema / 40,
            },
        },
        "ProvenanceCompletenessRate": {
            "completed_outputs": {
                "count": provenance_complete,
                "denominator": completed,
                "rate": provenance_complete / completed,
            },
            "end_to_end": {
                "count": provenance_complete,
                "denominator": 40,
                "rate": provenance_complete / 40,
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
        "provider_semantic_bodies_fully_empty": provider_bodies_empty,
        "raw_regulated_entity_proposals": raw_entities,
        "final_regulated_entities_accepted": final_entities,
        "raw_rule_proposals": raw_rules,
        "final_rules_accepted": final_rules,
        "diagnostic_code_counts": dict(sorted(all_diagnostics.items())),
    }
    result_set_hash = hashlib.sha256(
        b"".join(
            record_id.encode("utf-8") + payload for record_id, payload in sorted(raw_result_bytes)
        )
    ).hexdigest()
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "phase11_hybrid_stage_b2_clean_holdout_evaluation",
        "stage": "hybrid-qwen-v1-stage-b2",
        "split": "holdout",
        "record_count": 40,
        "completed_record_count": completed,
        "failed_record_count": len(failed_diagnostics),
        "reference_status": "AI-reviewed/adjudicated reference; not human gold",
        "selection_fingerprint": PHASE11_SELECTION_FINGERPRINT,
        "annotation_release_fingerprint": manifest["annotation_release_fingerprint"],
        "annotation_release_sha256": hashlib.sha256(HOLDOUT_RELEASE_PATH.read_bytes()).hexdigest(),
        "candidate_policy": "phase11-candidates-v3",
        "candidate_policy_hash": PHASE11_CANDIDATE_POLICY_HASH,
        "template_version": HYBRID_STAGE_B2_PROMPT_TEMPLATE_VERSION,
        "template_sha256": PROMPT_HASH,
        "schema_sha256": SCHEMA_HASH,
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "conditional_39_record_view": {
            key: value for key, value in conditional.items() if key != "per_record"
        },
        "end_to_end_40_record_view": {
            key: value for key, value in end_to_end.items() if key != "per_record"
        },
        "safety_structural_metrics": safety,
        "field_local_rejection_diagnostics": diagnostics_report,
        "error_taxonomy": _taxonomy(pairs, field_local),
        "complexity_breakdown": _complexity_breakdown(pairs),
        "failure_diagnostics": failed_diagnostics,
        "result_set_sha256": result_set_hash,
        "configuration_sha256": hashlib.sha256(
            HYBRID_STAGE_B2_HOLDOUT_CONFIG_PATH.read_bytes()
        ).hexdigest(),
        "selected_configuration_manifest_sha256": hashlib.sha256(
            SELECTED_CONFIG_PATH.read_bytes()
        ).hexdigest(),
        "method_notes": {
            "rule_matching": (
                "same strict rule-key matching and index-aligned modality "
                "comparison as DEV Stage B2"
            ),
            "failed_record_treatment": (
                "no predictions; gold contributes false negatives in the "
                "end-to-end view"
            ),
            "references": "AI-reviewed/adjudicated; not human gold",
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
        PRIVATE_EVALUATION_ROOT / "failed_record_diagnostic.json",
        {"failure_diagnostics": failed_diagnostics},
    )
    write_text_free_json(TRACKED_EVALUATION_PATH, report)
    report["tracked_evaluation_sha256"] = hashlib.sha256(
        TRACKED_EVALUATION_PATH.read_bytes()
    ).hexdigest()
    write_private_json(PRIVATE_EVALUATION_ROOT / "evaluation_report_private.json", report)
    return report


__all__ = ["evaluate_clean_stage_b2_holdout"]
