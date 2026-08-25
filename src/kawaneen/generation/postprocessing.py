"""Post-generation structural verification and conservative final rendering."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from kawaneen.generation.abstention import invalid_generation_result
from kawaneen.generation.contracts import (
    AbstentionReason,
    ClaimMode,
    GenerationDecision,
    GenerationResult,
    ModelOutputCitation,
    ModelOutputClaim,
    VerifiedClaim,
)
from kawaneen.generation.rendering import render_verified_answer
from kawaneen.generation.semantic import DeferredSemanticSupport, SemanticSupport
from kawaneen.grounding.citations import verify_citation
from kawaneen.grounding.contracts import (
    CitationRequest,
    ClaimDraft,
    ContextPack,
    GeneratedDraft,
    VerificationResult,
    VerifiedCitation,
)
from kawaneen.grounding.provenance import CanonicalCorpusResolver
from kawaneen.grounding.verification import verify_draft


class FinalizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result: GenerationResult
    verification: VerificationResult | None = None
    verified_claims: tuple[VerifiedClaim, ...] = ()
    rendered_answer: str | None = None

    @property
    def abstention_reason(self) -> AbstentionReason | None:
        return self.result.abstention_reason


def finalize_generation(
    pack: ContextPack,
    result: GenerationResult,
    resolver: CanonicalCorpusResolver,
    *,
    jurisdiction_text: str | None = None,
    disclaimer_text: str = "",
    semantic_support: SemanticSupport | None = None,
) -> FinalizationResult:
    if result.decision is GenerationDecision.ABSTAIN:
        return FinalizationResult(result=result)
    draft_claims: list[ClaimDraft] = []
    for index, claim in enumerate(result.claims, start=1):
        if claim.mode is ClaimMode.DIRECT:
            claim_text = "\n".join(citation.quoted_text for citation in claim.citations)
        elif claim.text is not None:
            claim_text = claim.text
        else:
            return FinalizationResult(
                result=invalid_generation_result("interpretation claim has no text")
            )
        draft_claims.append(
            ClaimDraft(
                claim_id=f"C{index:03d}",
                claim_text=claim_text,
                citations=tuple(
                    CitationRequest(
                        evidence_id=citation.evidence_id,
                        quoted_text=citation.quoted_text,
                    )
                    for citation in claim.citations
                ),
            )
        )
    draft = GeneratedDraft(
        answer_text="\n".join(claim.claim_text for claim in draft_claims),
        claims=tuple(draft_claims),
    )
    verification = verify_draft(pack, draft, resolver)
    if not verification.structurally_valid:
        return FinalizationResult(
            result=invalid_generation_result("Phase-9 structural citation verification failed"),
            verification=verification,
        )
    support = semantic_support or DeferredSemanticSupport()
    verified_claims: list[VerifiedClaim] = []
    for claim in result.claims:
        citations: list[VerifiedCitation] = []
        for citation in claim.citations:
            checked = verify_citation(
                pack,
                CitationRequest(
                    evidence_id=citation.evidence_id,
                    quoted_text=citation.quoted_text,
                ),
                resolver,
            )
            if not checked.valid or checked.citation is None:
                return FinalizationResult(
                    result=invalid_generation_result("citation verification failed"),
                    verification=verification,
                )
            citations.append(checked.citation)
        if claim.mode is ClaimMode.DIRECT:
            verified_claims.append(
                VerifiedClaim(
                    mode=ClaimMode.DIRECT,
                    text="\n".join(citation.quoted_text for citation in citations),
                    citations=tuple(citations),
                )
            )
            continue
        if claim.text is None:
            return FinalizationResult(
                result=invalid_generation_result("interpretation claim has no text"),
                verification=verification,
            )
        evidence_text = "\n".join(citation.quoted_text for citation in citations)
        assessment = support.assess(claim.text, evidence_text)
        if not assessment.available or assessment.supported is not True:
            continue
        verified_claims.append(
            VerifiedClaim(
                mode=ClaimMode.INTERPRETATION,
                text=claim.text,
                citations=tuple(citations),
            )
        )
    if not verified_claims:
        return FinalizationResult(
            result=GenerationResult(
                decision=GenerationDecision.ABSTAIN,
                abstention_reason=AbstentionReason.SEMANTIC_SUPPORT_UNAVAILABLE,
                detail="interpretation claim support is unavailable",
            ),
            verification=verification,
        )
    accepted_claims = tuple(
        ModelOutputClaim(
            mode=claim.mode,
            text=claim.text,
            citations=tuple(
                ModelOutputCitation(
                    evidence_id=citation.evidence_id,
                    quoted_text=citation.quoted_text,
                )
                for citation in claim.citations
            ),
        )
        for claim in verified_claims
    )
    accepted_result = GenerationResult(
        decision=GenerationDecision.ANSWER,
        claims=accepted_claims,
    )
    claims_tuple = tuple(verified_claims)
    return FinalizationResult(
        result=accepted_result,
        verification=verification,
        verified_claims=claims_tuple,
        rendered_answer=render_verified_answer(
            claims_tuple,
            jurisdiction_text=jurisdiction_text,
            disclaimer_text=disclaimer_text,
        ),
    )
