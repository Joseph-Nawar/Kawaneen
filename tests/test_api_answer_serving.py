from __future__ import annotations

from kawaneen.generation.contracts import (
    AbstentionReason,
)
from kawaneen.generation.policy import PolicyOutcome
from kawaneen.grounding.contracts import (
    CitationRequest,
    ClaimDraft,
    GeneratedDraft,
    VerificationResult,
    VerifiedCitation,
)
from kawaneen.retrieval.serving import (
    ServingEvidence,
    ServingRetrievalResult,
    ServingRetrievalSummary,
)


def _retrieval() -> ServingRetrievalResult:
    return ServingRetrievalResult(
        evidence=(
            ServingEvidence(
                chunk_id="chunk-1",
                rank=1,
                text="The deadline is thirty days.",
                document_id="doc-1",
                document_title="Regulation",
                article="1",
                page=None,
                source_url=None,
                score=2.5,
            ),
        ),
        summary=ServingRetrievalSummary(returned_count=1),
    )


def test_answer_policy_abstention_prevents_generation() -> None:
    from kawaneen.generation.serving import ServingAnswerer

    calls = 0

    def generate(query: str, context: object) -> GeneratedDraft:
        nonlocal calls
        calls += 1
        raise AssertionError("generation must not run after policy abstention")

    answerer = ServingAnswerer(
        retriever=lambda query, limit=8: _retrieval(),
        context_builder=lambda query, retrieval: object(),
        policy_evaluator=lambda query, context: PolicyOutcome(
            allowed=False,
            reason=AbstentionReason.PERSONALIZED_LEGAL_ADVICE,
            detail="refused",
        ),
        generator=generate,
        verifier=lambda context, draft: VerificationResult(
            valid_citations=(),
            invalid_citations=(),
            unsupported_claims=(),
            structurally_valid=False,
            should_abstain=True,
        ),
    )

    result = answerer.answer("Should I sign this?")

    assert result.answerable is False
    assert result.answer is None
    assert result.abstention_reason == AbstentionReason.PERSONALIZED_LEGAL_ADVICE.value
    assert calls == 0


def test_answer_returns_only_verified_citations() -> None:
    from kawaneen.generation.serving import ServingAnswerer

    draft = GeneratedDraft(
        answer_text="The deadline is thirty days.",
        claims=(
            ClaimDraft(
                claim_id="claim-1",
                claim_text="The deadline is thirty days.",
                citations=(CitationRequest(evidence_id="E001", quoted_text="thirty days"),),
            ),
        ),
    )
    verified = VerifiedCitation(
        evidence_id="E001",
        document_id="doc-1",
        document_title="Regulation",
        jurisdiction="SA",
        article="1",
        page=None,
        chunk_id="chunk-1",
        source_url=None,
        quoted_text="thirty days",
    )
    answerer = ServingAnswerer(
        retriever=lambda query, limit=8: _retrieval(),
        context_builder=lambda query, retrieval: object(),
        policy_evaluator=lambda query, context: PolicyOutcome(allowed=True),
        generator=lambda query, context: draft,
        verifier=lambda context, generated: VerificationResult(
            valid_citations=(verified,),
            invalid_citations=(),
            unsupported_claims=(),
            structurally_valid=True,
            should_abstain=False,
        ),
    )

    result = answerer.answer("What is the deadline?")

    assert result.answerable is True
    assert result.answer == draft.answer_text
    assert result.citations == (verified,)


def test_invalid_generation_or_citations_abstain_fail_closed() -> None:
    from kawaneen.generation.serving import ServingAnswerer

    answerer = ServingAnswerer(
        retriever=lambda query, limit=8: _retrieval(),
        context_builder=lambda query, retrieval: object(),
        policy_evaluator=lambda query, context: PolicyOutcome(allowed=True),
        generator=lambda query, context: GeneratedDraft(answer_text="unsupported", claims=()),
        verifier=lambda context, generated: VerificationResult(
            valid_citations=(),
            invalid_citations=(),
            unsupported_claims=("claim-1",),
            structurally_valid=False,
            should_abstain=True,
        ),
    )

    result = answerer.answer("What is the deadline?")

    assert result.answerable is False
    assert result.answer is None
    assert result.abstention_reason == AbstentionReason.INVALID_GENERATION.value
