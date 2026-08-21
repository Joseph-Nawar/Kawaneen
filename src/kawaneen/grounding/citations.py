"""Exact, context-local citation construction."""

from __future__ import annotations

from kawaneen.grounding.contracts import (
    CitationRequest,
    CitationVerification,
    ContextPack,
    VerifiedCitation,
)
from kawaneen.grounding.provenance import CanonicalCorpusResolver


def verify_citation(
    pack: ContextPack,
    request: CitationRequest,
    resolver: CanonicalCorpusResolver,
) -> CitationVerification:
    """Verify an exact codepoint quote and build metadata from canonical source."""

    evidence = next(
        (item for item in pack.evidence if item.evidence_id == request.evidence_id),
        None,
    )
    if evidence is None:
        return _invalid(request, "unknown evidence ID")
    if not request.quoted_text.strip():
        return _invalid(request, "quotation must be non-empty")

    candidates: list[tuple[int, str]] = []
    for chunk_id, rank in zip(
        evidence.contributing_chunk_ids, evidence.contributing_ranks, strict=True
    ):
        resolved = resolver.resolve_chunk(chunk_id)
        unit = next((row for row in resolved.units if row.unit_id == evidence.unit_id), None)
        if unit is not None and request.quoted_text in unit.display_text:
            candidates.append((rank, chunk_id))
    if not candidates:
        return _invalid(request, "quotation is not an exact authoritative substring")

    _, chunk_id = min(candidates, key=lambda item: (item[0], item[1]))
    canonical_unit = resolver.units_by_id[evidence.unit_id]
    citation = VerifiedCitation(
        evidence_id=evidence.evidence_id,
        document_id=canonical_unit.source.document_id,
        document_title=canonical_unit.source.document_title,
        jurisdiction=canonical_unit.source.jurisdiction,
        article=canonical_unit.source.article,
        page=canonical_unit.source.page,
        chunk_id=chunk_id,
        source_url=canonical_unit.source.source_url,
        quoted_text=request.quoted_text,
    )
    return CitationVerification(request=request, valid=True, citation=citation)


def _invalid(request: CitationRequest, reason: str) -> CitationVerification:
    return CitationVerification(request=request, valid=False, reason=reason)
