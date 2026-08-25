from __future__ import annotations

import pytest
from pydantic import ValidationError

from integration.test_query_to_answer import _write_resolver_inputs
from kawaneen.grounding.assembly import ContextAssembler
from kawaneen.grounding.citations import verify_citation
from kawaneen.grounding.contracts import CitationRequest, RetrievalInput

pytestmark = pytest.mark.integration


class _Counter:
    identity = "phase14"

    def count(self, text: str) -> int:
        return len(text)


def test_verified_citation_resolves_exact_canonical_metadata_and_quote(tmp_path) -> None:
    units, chunks, resolver = _write_resolver_inputs(tmp_path)
    pack = ContextAssembler(resolver, _Counter(), max_context_tokens=10_000).assemble(
        query_id="q1",
        ranked_inputs=(RetrievalInput(query_id="q1", rank=1, chunk_id=chunks[0].chunk_id),),
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
    )
    evidence = pack.evidence[0]

    result = verify_citation(
        pack,
        CitationRequest(evidence_id=evidence.evidence_id, quoted_text=evidence.display_text),
        resolver,
    )

    assert result.valid is True
    assert result.citation is not None
    assert result.citation.document_id == units[0].document_id
    assert result.citation.document_title == "Synthetic Appeals Regulation"
    assert result.citation.article == "المادة ١٢"
    assert result.citation.page is None
    assert result.citation.quoted_text == units[0].text


@pytest.mark.parametrize(
    "citation_request",
    [
        CitationRequest(evidence_id="E001", quoted_text="forged quote"),
        CitationRequest(evidence_id="E999", quoted_text="quote"),
    ],
)
def test_quote_and_evidence_mutations_fail_closed(
    tmp_path, citation_request: CitationRequest
) -> None:
    _, chunks, resolver = _write_resolver_inputs(tmp_path)
    pack = ContextAssembler(resolver, _Counter(), max_context_tokens=10_000).assemble(
        query_id="q1",
        ranked_inputs=(RetrievalInput(query_id="q1", rank=1, chunk_id=chunks[0].chunk_id),),
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
    )

    result = verify_citation(pack, citation_request, resolver)

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
