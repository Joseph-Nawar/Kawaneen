from __future__ import annotations

import pytest
from phase14_support import build_phase14_stack
from pydantic import ValidationError

from kawaneen.grounding.citations import verify_citation
from kawaneen.grounding.contracts import CitationRequest

pytestmark = pytest.mark.integration


def test_verified_citation_resolves_exact_canonical_metadata_and_quote() -> None:
    stack = build_phase14_stack()
    retrieval = stack.retriever.search("الاعتراض خلال ثلاثين يوماً")
    pack = stack.context_for("الاعتراض خلال ثلاثين يوماً", retrieval)
    evidence = pack.evidence[0]

    result = verify_citation(
        pack,
        CitationRequest(evidence_id=evidence.evidence_id, quoted_text=evidence.display_text),
        stack.resolver,
    )

    assert result.valid is True
    assert result.citation is not None
    assert result.citation.document_id == stack.units[0].document_id
    assert result.citation.document_title == "Synthetic Appeals Regulation"
    assert result.citation.article == "المادة ١٤"
    assert result.citation.page == "1"
    assert result.citation.quoted_text == next(
        unit.text for unit in stack.units if unit.ordinal == 14
    )


@pytest.mark.parametrize(
    "citation_request",
    [
        CitationRequest(evidence_id="E001", quoted_text="forged quote"),
        CitationRequest(evidence_id="E999", quoted_text="quote"),
    ],
)
def test_quote_and_evidence_mutations_fail_closed(
    citation_request: CitationRequest,
) -> None:
    stack = build_phase14_stack()
    retrieval = stack.retriever.search("الاعتراض خلال ثلاثين يوماً")
    pack = stack.context_for("الاعتراض خلال ثلاثين يوماً", retrieval)

    result = verify_citation(pack, citation_request, stack.resolver)

    assert result.valid is False
    assert result.citation is None


@pytest.mark.parametrize("field", ["document_id", "article", "page", "source_url"])
def test_generator_cannot_supply_mutable_source_metadata(field: str) -> None:
    with pytest.raises(ValidationError):
        CitationRequest(
            evidence_id="E001",
            quoted_text="exact source",
            **{field: "forged metadata"},
        )
