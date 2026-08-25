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
