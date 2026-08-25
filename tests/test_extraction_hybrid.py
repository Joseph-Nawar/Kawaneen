import json

import pytest

from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.contracts import (
    ProvenanceOrigin,
    SemanticProposal,
)
from kawaneen.extraction.deterministic import run_deterministic
from kawaneen.extraction.hybrid import assemble_hybrid_result, semantic_proposal_schema


def _base(text: str):
    return run_deterministic(
        text,
        canonical_unit_id="u1",
        document_id="d1",
        source_provenance=SourceProvenance(
            source_id="saudi-moj-derived",
            source_version="8",
            source_path="local",
            source_row=1,
            source_field="text",
        ),
    )


def test_provider_schema_contains_only_compact_source_grounded_fields() -> None:
    schema = semantic_proposal_schema()
    serialized = json.dumps(schema, ensure_ascii=False)
    assert "issuing_authority" not in serialized
    assert "document_id" not in serialized
    assert "start_char" not in serialized
    assert "candidate_registry" not in serialized


def test_hybrid_assembles_multiple_modalities_and_candidate_classifications() -> None:
    text = "يجب على المنشأة التسجيل خلال ٣٠ يوماً، ولا يجوز التأخير، ويجوز التمديد وفق المادة (7)."
    base = _base(text)
    proposal = {
        "schema_version": "phase11-proposal-v1",
        "regulated_entities": [{"text": "المنشأة"}],
        "rules": [
            {
                "modality": "obligation",
                "actor": {"text": "المنشأة"},
                "action": {"text": "التسجيل"},
                "deadline_refs": ["T001"],
            },
            {"modality": "prohibition", "action": {"text": "التأخير"}},
            {"modality": "permission", "action": {"text": "التمديد"}, "effective_date_refs": []},
        ],
        "exceptions": [],
        "penalties": [],
        "deadline_refs": ["T001"],
        "effective_date_refs": [],
        "monetary_threshold_refs": [],
        "percentage_threshold_refs": [],
    }
    result = assemble_hybrid_result(text, base, proposal)
    assert len(result.obligations) == 1
    assert len(result.prohibitions) == 1
    assert len(result.permissions) == 1
    assert result.deadlines[0].candidate_id == "T001"
    assert any(item.origin is ProvenanceOrigin.LLM_SELECTED for item in result.field_provenance)
    assert result.validation_metadata.raw_provider_schema_valid is True


def test_unsupported_and_ambiguous_spans_are_dropped_fail_closed() -> None:
    text = "يجوز التمديد."
    base = _base(text)
    proposal = {
        "schema_version": "phase11-proposal-v1",
        "regulated_entities": [{"text": "غير موجود"}],
        "rules": [
            {"modality": "permission", "action": {"text": "التمديد"}},
            {"modality": "permission", "action": {"text": "يجوز"}, "deadline_refs": ["T999"]},
        ],
        "exceptions": [],
        "penalties": [{"text": "غرامة"}],
    }
    result = assemble_hybrid_result(text, base, proposal)
    assert result.permissions[0].action_span.text == "التمديد"
    assert len(result.permissions) == 2
    assert result.permissions[1].deadline_refs == ()
    assert result.regulated_entities == ()
    assert result.penalties == ()
    assert result.validation_metadata.diagnostics
    assert all(not diagnostic.accepted for diagnostic in result.validation_metadata.diagnostics)


def test_malformed_provider_json_preserves_safe_deterministic_base() -> None:
    base = _base("المادة (7)")
    result = assemble_hybrid_result("المادة (7)", base, "not json")
    assert result.configuration == "hybrid-qwen-v1"
    assert result.referenced_articles[0].candidate_id == "A001"
    assert result.validation_metadata.raw_provider_schema_valid is False


def test_proposal_contract_forbids_metadata_and_offsets() -> None:
    with pytest.raises(ValueError):
        SemanticProposal.model_validate(
            {
                "schema_version": "phase11-proposal-v1",
                "rules": [
                    {
                        "modality": "obligation",
                        "action": {"text": "التسجيل", "document_id": "d1"},
                    }
                ],
            }
        )


def test_unique_span_with_invalid_occurrence_is_resolved_server_side() -> None:
    text = "يجوز تمديد المهلة."
    result = assemble_hybrid_result(
        text,
        _base(text),
        {
            "schema_version": "phase11-proposal-v1",
            "rules": [
                {"modality": "permission", "action": {"text": "تمديد المهلة", "occurrence": 7}}
            ],
        },
    )
    assert result.permissions[0].action_span.start_char == text.index("تمديد المهلة")
    assert result.validation_metadata.diagnostics[0].code == "INVALID_OCCURRENCE_CORRECTED"


def test_repeated_span_without_occurrence_is_rejected() -> None:
    text = "يجوز التمديد، ويجوز التمديد."
    result = assemble_hybrid_result(
        text,
        _base(text),
        {
            "schema_version": "phase11-proposal-v1",
            "rules": [{"modality": "permission", "action": {"text": "التمديد"}}],
        },
    )
    assert result.rules == ()
    assert result.validation_metadata.diagnostics[0].code == "AMBIGUOUS_OR_INVALID_OCCURRENCE"


def test_missing_registry_fails_closed_without_inventing_semantics() -> None:
    text = "يجوز التمديد."
    base = _base(text).model_copy(update={"candidate_registry": None})
    result = assemble_hybrid_result(
        text,
        base,
        {
            "schema_version": "phase11-proposal-v1",
            "regulated_entities": [{"text": "التمديد"}],
            "rules": [{"modality": "permission", "action": {"text": "التمديد"}}],
        },
    )
    assert result.regulated_entities == ()
    assert result.rules == ()
    assert all(item.code == "MISSING_REGISTRY" for item in result.validation_metadata.diagnostics)


def test_invalid_optional_spans_are_dropped_and_valid_fields_are_retained() -> None:
    text = "يجوز التمديد إذا تحقق الشرط، إلا في الاستثناء، والحد 1250 SAR ونسبة 15%."
    result = assemble_hybrid_result(
        text,
        _base(text),
        {
            "schema_version": "phase11-proposal-v1",
            "regulated_entities": [],
            "rules": [
                {
                    "modality": "permission",
                    "actor": {"text": "فاعل"},
                    "action": {"text": "التمديد"},
                    "conditions": [{"text": "تحقق الشرط"}, {"text": "غير موجود"}],
                    "exceptions": [{"text": "الاستثناء"}, {"text": "غير موجود"}],
                    "monetary_threshold_refs": ["M001", "P001"],
                    "percentage_threshold_refs": ["P001", "M001"],
                }
            ],
            "exceptions": [{"text": "الاستثناء"}],
            "penalties": [{"text": "الحد"}],
        },
    )
    rule = result.permissions[0]
    assert [span.text for span in rule.condition_spans] == ["تحقق الشرط"]
    assert [span.text for span in rule.exception_spans] == ["الاستثناء"]
    assert rule.monetary_threshold_refs == ("M001",)
    assert rule.percentage_threshold_refs == ("P001",)
    assert result.exceptions[0].text == "الاستثناء"
    assert result.penalties[0].text == "الحد"


def test_empty_action_drops_only_the_invalid_rule() -> None:
    text = "يجوز التمديد."
    result = assemble_hybrid_result(
        text,
        _base(text),
        {
            "schema_version": "phase11-proposal-v1",
            "rules": [{"modality": "permission", "action": {"text": ""}}],
        },
    )
    assert result.rules == ()
    assert result.validation_metadata.diagnostics[0].code == "INVALID_PROVIDER_JSON"


def test_top_level_invalid_spans_and_candidate_refs_are_dropped() -> None:
    text = "يجوز التمديد خلال 30 يوماً."
    result = assemble_hybrid_result(
        text,
        _base(text),
        {
            "schema_version": "phase11-proposal-v1",
            "exceptions": [{"text": "غير موجود"}],
            "deadline_refs": ["T001", "M001"],
            "monetary_threshold_refs": ["M999"],
            "percentage_threshold_refs": ["P999"],
        },
    )
    assert [candidate.candidate_id for candidate in result.deadlines] == ["T001"]
    assert result.monetary_thresholds == ()
    assert result.percentage_thresholds == ()
    assert result.exceptions == ()
