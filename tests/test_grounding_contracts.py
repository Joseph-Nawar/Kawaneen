from __future__ import annotations

import pytest
from pydantic import ValidationError

from kawaneen.grounding.contracts import (
    CanonicalEvidenceUnit,
    CitationRequest,
    ClaimDraft,
    ContextPack,
    GeneratedDraft,
    SourceRecord,
)


def source() -> SourceRecord:
    return SourceRecord(document_id="doc-1", source_id="alarb")


def unit() -> CanonicalEvidenceUnit:
    return CanonicalEvidenceUnit(
        unit_id="unit-1",
        document_id="doc-1",
        ordinal=1,
        display_text="النص الأصلي",
        heading_path=(),
        source=source(),
    )


def test_source_record_is_frozen_and_forbids_unknown_metadata() -> None:
    value = source()
    with pytest.raises(ValidationError):
        SourceRecord(document_id="doc-1", fabricated_page=4)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        value.document_id = "other"  # type: ignore[misc]


def test_citation_request_exposes_only_evidence_id_and_quote() -> None:
    request = CitationRequest(evidence_id="E001", quoted_text="النص الأصلي")
    assert request.model_dump() == {
        "evidence_id": "E001",
        "quoted_text": "النص الأصلي",
    }
    with pytest.raises(ValidationError):
        CitationRequest(
            evidence_id="E001",
            quoted_text="النص الأصلي",
            document_id="doc-1",  # type: ignore[call-arg]
        )


def test_claim_and_draft_are_strict_but_allow_structural_validation_later() -> None:
    claim = ClaimDraft(claim_id="C001", claim_text="يثبت النص الواقعة", citations=())
    draft = GeneratedDraft(answer_text="إجابة", claims=(claim,))
    assert draft.claims[0].claim_id == "C001"
    assert ContextPack(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
        assembly_policy_version="phase9-v1",
        token_counter_identity="fake-v1",
        max_context_tokens=10,
        token_count=0,
        units=(),
        blocks=(),
        evidence=(),
        omissions=(),
    ).evidence == ()


def test_empty_identifiers_and_quotes_are_rejected_at_contract_boundary() -> None:
    with pytest.raises(ValidationError):
        CitationRequest(evidence_id="", quoted_text="quote")
    with pytest.raises(ValidationError):
        CanonicalEvidenceUnit(
            unit_id="unit-1",
            document_id="doc-1",
            ordinal=1,
            display_text="",
            heading_path=(),
            source=source(),
        )
