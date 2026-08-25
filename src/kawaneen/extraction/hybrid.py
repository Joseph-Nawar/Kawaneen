"""Hybrid deterministic/semantic assembly with fail-closed validation."""

from __future__ import annotations

import json

from kawaneen.extraction.contracts import (
    Candidate,
    CandidateType,
    ExactSourceSpan,
    ExtractionResult,
    FieldProvenance,
    Modality,
    NormativeRule,
    ProposedRule,
    ProposedSpan,
    ProvenanceOrigin,
    ValidationDiagnostic,
    ValidationMetadata,
)
from kawaneen.extraction.provider import parse_semantic_proposal, semantic_proposal_schema


def _diagnostic(
    code: str, field_name: str, message: str, *, accepted: bool = False
) -> ValidationDiagnostic:
    return ValidationDiagnostic(
        code=code, field_name=field_name, message=message, accepted=accepted
    )


def _resolve(
    canonical_text: str,
    proposed: ProposedSpan,
    base: ExtractionResult,
    field_name: str,
) -> tuple[ExactSourceSpan | None, ValidationDiagnostic | None]:
    registry = base.candidate_registry
    if registry is None:
        return None, _diagnostic(
            "MISSING_REGISTRY", field_name, "candidate registry is unavailable"
        )
    if not proposed.text:
        return None, _diagnostic(
            "UNSUPPORTED_MODEL_SPAN", field_name, "proposed source span is empty"
        )
    offsets: list[int] = []
    cursor = 0
    while True:
        found = canonical_text.find(proposed.text, cursor)
        if found < 0:
            break
        offsets.append(found)
        cursor = found + 1
    if not offsets:
        return None, _diagnostic(
            "UNSUPPORTED_MODEL_SPAN",
            field_name,
            "proposed source span not found in canonical text",
        )
    if len(offsets) == 1:
        selected = offsets[0]
        if proposed.occurrence not in (None, 0):
            return (
                ExactSourceSpan(
                    text=proposed.text,
                    start_char=selected,
                    end_char=selected + len(proposed.text),
                    canonical_unit_id=registry.canonical_unit_id,
                    document_id=registry.document_id,
                ),
                _diagnostic(
                    "INVALID_OCCURRENCE_CORRECTED",
                    field_name,
                    (
                        f"supplied occurrence={proposed.occurrence}; "
                        "unique exact span resolved at occurrence=0"
                    ),
                    accepted=True,
                ),
            )
    elif proposed.occurrence is None:
        return None, _diagnostic(
            "AMBIGUOUS_OR_INVALID_OCCURRENCE",
            field_name,
            "exact source span occurs multiple times and occurrence is missing",
        )
    elif proposed.occurrence < 0 or proposed.occurrence >= len(offsets):
        return None, _diagnostic(
            "AMBIGUOUS_OR_INVALID_OCCURRENCE",
            field_name,
            f"supplied occurrence={proposed.occurrence} does not resolve to an exact occurrence",
        )
    selected = offsets[0 if proposed.occurrence is None else proposed.occurrence]
    return (
        ExactSourceSpan(
            text=proposed.text,
            start_char=selected,
            end_char=selected + len(proposed.text),
            canonical_unit_id=registry.canonical_unit_id,
            document_id=registry.document_id,
        ),
        None,
    )


def _candidate(
    base: ExtractionResult,
    candidate_id: str,
    expected: CandidateType,
    field_name: str,
) -> tuple[Candidate | None, ValidationDiagnostic | None]:
    registry = base.candidate_registry
    if registry is None:
        return None, _diagnostic(
            "MISSING_REGISTRY", field_name, "candidate registry is unavailable"
        )
    candidate = next(
        (item for item in registry.candidates if item.candidate_id == candidate_id), None
    )
    if candidate is None or candidate.candidate_type is not expected:
        return None, _diagnostic(
            "INVALID_CANDIDATE_REFERENCE",
            field_name,
            f"rejected_id={candidate_id}; expected {expected.value} candidate",
        )
    return candidate, None


def _resolve_rule(
    canonical_text: str,
    proposed: ProposedRule,
    base: ExtractionResult,
    diagnostics: list[ValidationDiagnostic],
    index: int,
) -> NormativeRule | None:
    action, error = _resolve(canonical_text, proposed.action, base, f"rules[{index}].action")
    if error is not None:
        diagnostics.append(error)
    if action is None:
        return None
    actor = None
    if proposed.actor is not None:
        actor, error = _resolve(canonical_text, proposed.actor, base, f"rules[{index}].actor")
        if error is not None:
            diagnostics.append(error)
    conditions: list[ExactSourceSpan] = []
    for condition_index, condition in enumerate(proposed.conditions):
        resolved, error = _resolve(
            canonical_text, condition, base, f"rules[{index}].conditions[{condition_index}]"
        )
        if error is not None:
            diagnostics.append(error)
        if resolved is not None:
            conditions.append(resolved)
    exceptions: list[ExactSourceSpan] = []
    for exception_index, exception in enumerate(proposed.exceptions):
        resolved, error = _resolve(
            canonical_text, exception, base, f"rules[{index}].exceptions[{exception_index}]"
        )
        if error is not None:
            diagnostics.append(error)
        if resolved is not None:
            exceptions.append(resolved)
    refs: dict[str, list[str]] = {
        "deadline_refs": [],
        "monetary_threshold_refs": [],
        "percentage_threshold_refs": [],
    }
    for field_name, values, expected in (
        ("deadline_refs", proposed.deadline_refs, CandidateType.TEMPORAL),
        ("monetary_threshold_refs", proposed.monetary_threshold_refs, CandidateType.MONETARY),
        ("percentage_threshold_refs", proposed.percentage_threshold_refs, CandidateType.PERCENTAGE),
    ):
        for candidate_id in values:
            candidate, error = _candidate(
                base, candidate_id, expected, f"rules[{index}].{field_name}"
            )
            if error is not None:
                diagnostics.append(error)
            if candidate is not None:
                refs[field_name].append(candidate.candidate_id)
    return NormativeRule(
        modality=proposed.modality,
        actor_span=actor,
        action_span=action,
        condition_spans=tuple(conditions),
        exception_spans=tuple(exceptions),
        deadline_refs=tuple(refs["deadline_refs"]),
        monetary_threshold_refs=tuple(refs["monetary_threshold_refs"]),
        percentage_threshold_refs=tuple(refs["percentage_threshold_refs"]),
    )


def assemble_hybrid_result(
    canonical_text: str,
    base_result: ExtractionResult,
    raw_proposal: object,
) -> ExtractionResult:
    diagnostics: list[ValidationDiagnostic] = []
    try:
        proposal = parse_semantic_proposal(raw_proposal)
    except (TypeError, ValueError, json.JSONDecodeError) as error:  # type: ignore[name-defined]
        diagnostics.append(_diagnostic("INVALID_PROVIDER_JSON", "provider", str(error)))
        return base_result.model_copy(
            update={
                "configuration": "hybrid-qwen-v1",
                "extractor_version": "hybrid-qwen-v1",
                "validation_metadata": ValidationMetadata(
                    raw_provider_schema_valid=False,
                    proposal_valid=False,
                    diagnostics=tuple(diagnostics),
                ),
            }
        )
    entities: list[ExactSourceSpan] = []
    for index, proposed in enumerate(proposal.regulated_entities):
        resolved, error = _resolve(
            canonical_text, proposed, base_result, f"regulated_entities[{index}]"
        )
        if error is not None:
            diagnostics.append(error)
        if resolved is not None:
            entities.append(resolved)
    exceptions: list[ExactSourceSpan] = []
    for index, proposed in enumerate(proposal.exceptions):
        resolved, error = _resolve(canonical_text, proposed, base_result, f"exceptions[{index}]")
        if error is not None:
            diagnostics.append(error)
        if resolved is not None:
            exceptions.append(resolved)
    penalties: list[ExactSourceSpan] = []
    for index, proposed in enumerate(proposal.penalties):
        resolved, error = _resolve(canonical_text, proposed, base_result, f"penalties[{index}]")
        if error is not None:
            diagnostics.append(error)
        if resolved is not None:
            penalties.append(resolved)
    rules = tuple(
        rule
        for index, proposed in enumerate(proposal.rules)
        if (rule := _resolve_rule(canonical_text, proposed, base_result, diagnostics, index))
        is not None
    )

    def refs(
        values: tuple[str, ...], expected: CandidateType, field_name: str
    ) -> tuple[Candidate, ...]:
        selected: list[Candidate] = []
        for candidate_id in values:
            candidate, error = _candidate(base_result, candidate_id, expected, field_name)
            if error is not None:
                diagnostics.append(error)
            if candidate is not None:
                selected.append(candidate)
        return tuple(selected)

    deadlines = refs(proposal.deadline_refs, CandidateType.TEMPORAL, "deadline_refs")
    effective_dates = refs(
        proposal.effective_date_refs, CandidateType.TEMPORAL, "effective_date_refs"
    )
    monetary = refs(
        proposal.monetary_threshold_refs, CandidateType.MONETARY, "monetary_threshold_refs"
    )
    percentages = refs(
        proposal.percentage_threshold_refs, CandidateType.PERCENTAGE, "percentage_threshold_refs"
    )
    obligations = tuple(rule for rule in rules if rule.modality is Modality.OBLIGATION)
    prohibitions = tuple(rule for rule in rules if rule.modality is Modality.PROHIBITION)
    permissions = tuple(rule for rule in rules if rule.modality is Modality.PERMISSION)
    provenance = (
        *base_result.field_provenance,
        FieldProvenance(field_name="regulated_entities", origin=ProvenanceOrigin.LLM_SELECTED),
        FieldProvenance(field_name="rules", origin=ProvenanceOrigin.LLM_SELECTED),
        FieldProvenance(field_name="deadlines", origin=ProvenanceOrigin.LLM_SELECTED),
        FieldProvenance(field_name="effective_dates", origin=ProvenanceOrigin.LLM_SELECTED),
        FieldProvenance(field_name="monetary_thresholds", origin=ProvenanceOrigin.LLM_SELECTED),
        FieldProvenance(field_name="percentage_thresholds", origin=ProvenanceOrigin.LLM_SELECTED),
        FieldProvenance(field_name="exceptions", origin=ProvenanceOrigin.LLM_SELECTED),
        FieldProvenance(field_name="penalties", origin=ProvenanceOrigin.LLM_SELECTED),
    )
    return base_result.model_copy(
        update={
            "configuration": "hybrid-qwen-v1",
            "extractor_version": "hybrid-qwen-v1",
            "regulated_entities": tuple(entities),
            "rules": obligations + prohibitions + permissions,
            "obligations": obligations,
            "prohibitions": prohibitions,
            "permissions": permissions,
            "deadlines": deadlines,
            "effective_dates": effective_dates,
            "monetary_thresholds": monetary,
            "percentage_thresholds": percentages,
            "exceptions": tuple(exceptions),
            "penalties": tuple(penalties),
            "field_provenance": provenance,
            "validation_metadata": ValidationMetadata(
                raw_provider_schema_valid=True,
                proposal_valid=True,
                diagnostics=tuple(diagnostics),
            ),
        }
    )


__all__ = ["assemble_hybrid_result", "semantic_proposal_schema"]
