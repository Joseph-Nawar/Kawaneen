"""Structural draft verification; semantic entailment is deferred to Phase 10."""

from __future__ import annotations

from kawaneen.grounding.citations import verify_citation
from kawaneen.grounding.contracts import (
    CitationRequest,
    CitationVerification,
    ContextPack,
    GeneratedDraft,
    VerificationResult,
    VerifiedCitation,
)
from kawaneen.grounding.provenance import CanonicalCorpusResolver


def verify_draft(
    pack: ContextPack,
    draft: GeneratedDraft,
    resolver: CanonicalCorpusResolver,
) -> VerificationResult:
    """Apply conservative structural grounding and abstention rules only."""

    invalid: list[CitationVerification] = []
    valid: list[VerifiedCitation] = []
    unsupported: list[str] = []
    substantive = bool(draft.answer_text.strip())
    if not pack.evidence:
        return VerificationResult(
            valid_citations=(),
            invalid_citations=(),
            unsupported_claims=tuple(claim.claim_id for claim in draft.claims),
            structurally_valid=False,
            should_abstain=True,
        )
    if substantive and not draft.claims:
        return VerificationResult(
            valid_citations=(),
            invalid_citations=(),
            unsupported_claims=(),
            structurally_valid=False,
            should_abstain=True,
        )

    for claim in draft.claims:
        represented = claim.claim_text in draft.answer_text
        claim_valid_citations: list[VerifiedCitation] = []
        for request in claim.citations:
            result = verify_citation(pack, request, resolver)
            if result.valid and result.citation is not None:
                valid.append(result.citation)
                claim_valid_citations.append(result.citation)
            else:
                invalid.append(result)
        if not represented or not claim_valid_citations:
            unsupported.append(claim.claim_id)

    structurally_valid = not unsupported and not invalid
    return VerificationResult(
        valid_citations=tuple(valid),
        invalid_citations=tuple(invalid),
        unsupported_claims=tuple(unsupported),
        structurally_valid=structurally_valid,
        should_abstain=not structurally_valid,
    )


def citation_requests(draft: GeneratedDraft) -> tuple[CitationRequest, ...]:
    """Return generator-facing requests in deterministic claim order."""

    return tuple(request for claim in draft.claims for request in claim.citations)
