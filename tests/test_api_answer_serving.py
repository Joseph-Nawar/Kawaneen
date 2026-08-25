from __future__ import annotations

import pytest

from kawaneen.generation.contracts import (
    AbstentionReason,
    GenerationDecision,
    GenerationResult,
    ModelOutputCitation,
    ModelOutputClaim,
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


def test_stage_d_serving_generator_resolves_request_local_quote_registry() -> None:
    from kawaneen.generation.serving import StageDServingGenerator
    from kawaneen.grounding.contracts import (
        ContextBlock,
        ContextPack,
        ContextUnit,
        EvidenceReference,
        SourceRecord,
    )

    source = SourceRecord(document_id="doc-1", source_id="fixture")
    unit = ContextUnit(
        unit_id="unit-1",
        document_id="doc-1",
        ordinal=1,
        display_text="النص القانوني",
        source=source,
        best_retrieval_rank=1,
        contributing_chunk_ids=("chunk-1",),
        contributing_ranks=(1,),
    )
    pack = ContextPack(
        query_id="query-1",
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
        assembly_policy_version="phase9-test",
        token_counter_identity="fixture",
        max_context_tokens=100,
        token_count=10,
        units=(unit,),
        blocks=(
            ContextBlock(
                block_id="B001",
                document_id="doc-1",
                source=source,
                units=(unit,),
                best_retrieval_rank=1,
            ),
        ),
        evidence=(
            EvidenceReference(
                evidence_id="E001",
                unit_id="unit-1",
                block_id="B001",
                document_id="doc-1",
                display_text="النص القانوني",
                source=source,
                contributing_chunk_ids=("chunk-1",),
                contributing_ranks=(1,),
            ),
        ),
        omissions=(),
        input_chunk_ids=("chunk-1",),
    )

    class FakeProvider:
        def generate(self, request: object) -> GenerationResult:
            assert request.quote_registry.entries[0].quote_id == "Q001"
            return GenerationResult(
                decision=GenerationDecision.ANSWER,
                claims=(
                    ModelOutputClaim(
                        citations=(
                            ModelOutputCitation(evidence_id="E001", quoted_text="النص القانوني"),
                        ),
                    ),
                ),
            )

    draft = StageDServingGenerator(FakeProvider())("query", pack)

    assert draft is not None
    assert draft.answer_text == "النص القانوني"
    assert draft.claims[0].citations[0].evidence_id == "E001"

    class AbstainingProvider:
        def generate(self, request: object) -> GenerationResult:
            return GenerationResult(decision=GenerationDecision.ABSTAIN)

    assert StageDServingGenerator(AbstainingProvider())("query", pack) is None

    class InvalidProvider:
        def generate(self, request: object) -> GenerationResult:
            return GenerationResult(
                decision=GenerationDecision.ABSTAIN,
                abstention_reason=AbstentionReason.INVALID_GENERATION,
            )

    invalid = StageDServingGenerator(InvalidProvider())("query", pack)
    assert invalid is not None
    assert invalid.answer_text == ""

    with pytest.raises(RuntimeError, match="provider"):
        StageDServingGenerator(object())("query", pack)


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
