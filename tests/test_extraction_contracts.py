import pytest
from pydantic import ValidationError

from kawaneen.extraction.contracts import (
    ExactSourceSpan,
    ExtractionResult,
    Modality,
    SemanticProposal,
)


def test_exact_source_span_rejects_empty_or_reversed_ranges() -> None:
    with pytest.raises(ValidationError):
        ExactSourceSpan(
            text="نص",
            start_char=3,
            end_char=3,
            canonical_unit_id="u1",
            document_id="d1",
        )


def test_semantic_proposal_forbids_metadata_and_arbitrary_fields() -> None:
    with pytest.raises(ValidationError):
        SemanticProposal.model_validate(
            {
                "schema_version": "phase11-proposal-v1",
                "issuing_authority": "وزارة العدل",
            }
        )


def test_extraction_result_has_required_grouped_schema() -> None:
    result = ExtractionResult(
        schema_version="phase11-extraction-v1",
        extractor_version="deterministic-v1",
        configuration="deterministic-v1",
        jurisdiction="SA",
        source_provenance={
            "source_id": "saudi-moj-derived",
            "source_version": "8",
            "source_path": "local",
            "source_row": 1,
            "source_field": "text",
        },
        source_fingerprint="a" * 64,
        issuing_authority=None,
    )
    assert result.obligations == ()
    assert result.prohibitions == ()
    assert result.permissions == ()
    assert result.jurisdiction == "SA"


def test_modality_values_are_locked() -> None:
    assert {item.value for item in Modality} == {"obligation", "prohibition", "permission"}
