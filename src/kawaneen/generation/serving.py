"""Single-query serving pipeline for grounded answers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from kawaneen.generation.answerability import SourceEligibility, evaluate_stage_d_policy
from kawaneen.generation.contracts import AbstentionReason
from kawaneen.generation.policy import JurisdictionScope, PolicyContext, PolicyOutcome
from kawaneen.grounding.assembly import ContextAssembler
from kawaneen.grounding.contracts import (
    GeneratedDraft,
    RetrievalInput,
    VerificationResult,
    VerifiedCitation,
)
from kawaneen.grounding.provenance import CanonicalCorpusResolver
from kawaneen.grounding.verification import verify_draft
from kawaneen.retrieval.serving import ServingRetrievalResult


class RetrievalService(Protocol):
    def __call__(self, query: str, limit: int = 8) -> ServingRetrievalResult: ...


class ContextBuilder(Protocol):
    def __call__(self, query: str, retrieval: ServingRetrievalResult) -> object: ...


class PolicyEvaluator(Protocol):
    def __call__(self, query: str, context: object) -> PolicyOutcome: ...


class DraftGenerator(Protocol):
    def __call__(self, query: str, context: object) -> GeneratedDraft: ...


class DraftVerifier(Protocol):
    def __call__(self, context: object, draft: GeneratedDraft) -> VerificationResult: ...


@dataclass(frozen=True, slots=True)
class ServingAnswerResult:
    answerable: bool
    answer: str | None
    abstention_reason: str | None
    citations: tuple[VerifiedCitation, ...]
    retrieval: ServingRetrievalResult
    warnings: tuple[str, ...] = ()


class GenerationModelUnavailableError(RuntimeError):
    """The configured Stage-D provider cannot serve this request."""


class ServingAnswerer:
    """Fail-closed one-query pipeline with policy before generation."""

    def __init__(
        self,
        *,
        retriever: RetrievalService,
        context_builder: ContextBuilder,
        policy_evaluator: PolicyEvaluator,
        generator: DraftGenerator,
        verifier: DraftVerifier,
    ) -> None:
        self.retriever = retriever
        self.context_builder = context_builder
        self.policy_evaluator = policy_evaluator
        self.generator = generator
        self.verifier = verifier

    @classmethod
    def from_phase9_10(
        cls,
        *,
        retriever: object,
        assembler: ContextAssembler,
        resolver: CanonicalCorpusResolver,
        phase8_selection_sha256: str,
        canonical_corpus_hash: str,
        deployment_scope: JurisdictionScope,
        generator: DraftGenerator,
        source_registry: Mapping[str, SourceEligibility] = {},
        structural_roles: Mapping[str, str] = {},
    ) -> ServingAnswerer:
        """Bind the frozen grounding and Stage-D policy primitives for serving."""

        def build_context(query: str, retrieval: ServingRetrievalResult) -> object:
            query_id = hashlib.sha256(query.encode("utf-8")).hexdigest()
            ranked = tuple(
                RetrievalInput(query_id=query_id, rank=item.rank, chunk_id=item.chunk_id)
                for item in retrieval.evidence
            )
            return assembler.assemble(
                query_id=query_id,
                ranked_inputs=ranked,
                phase8_selection_sha256=phase8_selection_sha256,
                canonical_corpus_hash=canonical_corpus_hash,
            )

        def policy(query: str, context: object) -> PolicyOutcome:
            return evaluate_stage_d_policy(
                query,
                PolicyContext(
                    context_pack=context,  # type: ignore[arg-type]
                    scope=deployment_scope,
                ),
                source_registry=source_registry,
                structural_roles=structural_roles,
            )

        def verify(context: object, draft: GeneratedDraft) -> VerificationResult:
            return verify_draft(context, draft, resolver)  # type: ignore[arg-type]

        return cls(
            retriever=callable_retriever(retriever),
            context_builder=build_context,
            policy_evaluator=policy,
            generator=generator,
            verifier=verify,
        )

    def answer(self, query: str) -> ServingAnswerResult:
        retrieval = self.retriever(query, 8)
        context = self.context_builder(query, retrieval)
        policy = self.policy_evaluator(query, context)
        if not policy.allowed:
            return ServingAnswerResult(
                answerable=False,
                answer=None,
                abstention_reason=(
                    policy.reason.value if policy.reason is not None else "REQUESTED_INFO_NOT_FOUND"
                ),
                citations=(),
                retrieval=retrieval,
            )
        try:
            draft = self.generator(query, context)
            verification = self.verifier(context, draft)
        except (TypeError, ValueError, KeyError, AttributeError):
            return ServingAnswerResult(
                answerable=False,
                answer=None,
                abstention_reason=AbstentionReason.INVALID_GENERATION.value,
                citations=(),
                retrieval=retrieval,
                warnings=("generation_or_verification_failed_closed",),
            )
        if verification.should_abstain or not verification.structurally_valid:
            return ServingAnswerResult(
                answerable=False,
                answer=None,
                abstention_reason=AbstentionReason.INVALID_GENERATION.value,
                citations=(),
                retrieval=retrieval,
                warnings=("citation_verification_failed_closed",),
            )
        return ServingAnswerResult(
            answerable=True,
            answer=draft.answer_text,
            abstention_reason=None,
            citations=verification.valid_citations,
            retrieval=retrieval,
        )


def callable_retriever(service: object) -> RetrievalService:
    """Adapt an object exposing ``search`` without importing the HTTP layer."""

    if callable(service):
        return cast(RetrievalService, service)
    search = getattr(service, "search", None)
    if not callable(search):
        raise TypeError("serving retriever must be callable or expose search")
    return cast(RetrievalService, search)


__all__ = ["ServingAnswerResult", "ServingAnswerer"]
