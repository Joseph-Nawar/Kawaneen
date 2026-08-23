from __future__ import annotations

from kawaneen.generation.contracts import AbstentionReason
from kawaneen.generation.policy import (
    JurisdictionScope,
    PolicyContext,
    default_deployment_scope,
    evaluate_pre_generation_policy,
)
from kawaneen.generation.semantic import DeferredSemanticSupport
from kawaneen.grounding.contracts import ContextPack, EvidenceReference, SourceRecord


def empty_pack() -> ContextPack:
    source = SourceRecord(document_id="doc-1", document_title="Law")
    return ContextPack(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
        assembly_policy_version="phase9-v1",
        token_counter_identity="fake-v1",
        max_context_tokens=100,
        token_count=0,
        units=(),
        blocks=(),
        evidence=(
            EvidenceReference(
                evidence_id="E001",
                unit_id="u1",
                block_id="B001",
                document_id="doc-1",
                display_text="A source sentence.",
                source=source,
                contributing_chunk_ids=("c1",),
                contributing_ranks=(1,),
            ),
        ),
        omissions=(),
    )


def context(**kwargs: object) -> PolicyContext:
    return PolicyContext(context_pack=empty_pack(), **kwargs)


def test_personalized_advice_is_refused_in_english_and_arabic() -> None:
    english = evaluate_pre_generation_policy(
        "Should I sue my employer?", context(scope=JurisdictionScope())
    )
    arabic = evaluate_pre_generation_policy(
        "هل يجب أن أرفع دعوى؟", context(scope=JurisdictionScope())
    )

    assert english.reason is AbstentionReason.PERSONALIZED_LEGAL_ADVICE
    assert arabic.reason is AbstentionReason.PERSONALIZED_LEGAL_ADVICE
    assert not english.allowed
    assert not arabic.allowed


def test_authoritative_jurisdiction_is_server_controlled_and_mismatch_fails() -> None:
    scope = JurisdictionScope(
        active_jurisdiction="SA",
        allowed_jurisdictions=("SA",),
        mode="single",
    )

    in_scope = evaluate_pre_generation_policy("What is the rule?", context(scope=scope))
    out_of_scope = evaluate_pre_generation_policy(
        "What is the rule in Egypt?", context(scope=scope)
    )

    assert in_scope.allowed
    assert in_scope.jurisdiction_text == "SA"
    assert out_of_scope.reason is AbstentionReason.JURISDICTION_MISMATCH
    assert not out_of_scope.allowed


def test_governed_default_scope_is_single_saudi_v1() -> None:
    scope = default_deployment_scope()

    assert scope.active_jurisdiction == "SA"
    assert scope.allowed_jurisdictions == ("SA",)
    assert scope.mode == "single"
    assert scope.authoritative_jurisdiction == "SA"


def test_english_egypt_request_fails_against_governed_saudi_scope() -> None:
    result = evaluate_pre_generation_policy(
        "What is the rule in Egypt?",
        context(scope=default_deployment_scope()),
    )

    assert result.reason is AbstentionReason.JURISDICTION_MISMATCH
    assert result.allowed is False


def test_arabic_egypt_request_fails_against_governed_saudi_scope() -> None:
    result = evaluate_pre_generation_policy(
        "ما الحكم في مصر؟",
        context(scope=default_deployment_scope()),
    )

    assert result.reason is AbstentionReason.JURISDICTION_MISMATCH
    assert result.allowed is False


def test_conflicting_english_jurisdictions_are_ambiguous() -> None:
    result = evaluate_pre_generation_policy(
        "Compare Saudi Arabia and Egypt for this rule.",
        context(scope=default_deployment_scope()),
    )

    assert result.reason is AbstentionReason.JURISDICTION_AMBIGUOUS
    assert result.allowed is False


def test_conflicting_arabic_jurisdictions_are_ambiguous() -> None:
    result = evaluate_pre_generation_policy(
        "قارن بين الحكم في السعودية ومصر.",
        context(scope=default_deployment_scope()),
    )

    assert result.reason is AbstentionReason.JURISDICTION_AMBIGUOUS
    assert result.allowed is False


def test_required_unresolved_scope_abstains() -> None:
    result = evaluate_pre_generation_policy(
        "What is the rule?",
        context(scope=JurisdictionScope(required=True)),
    )

    assert result.reason is AbstentionReason.JURISDICTION_AMBIGUOUS
    assert result.allowed is False


def test_explicit_known_jurisdiction_fails_closed_when_scope_is_unverified() -> None:
    result = evaluate_pre_generation_policy(
        "What is the rule in Egypt?",
        context(scope=JurisdictionScope()),
    )

    assert result.reason is AbstentionReason.JURISDICTION_MISMATCH
    assert result.allowed is False


def test_missing_scope_is_unverified_and_currentness_question_abstains() -> None:
    result = evaluate_pre_generation_policy(
        "Is this law currently in force?",
        context(scope=JurisdictionScope(), source_status_available=False),
    )

    assert result.allowed is False
    assert result.jurisdiction_verified is False
    assert result.reason is AbstentionReason.CURRENTNESS_UNVERIFIED


def test_superseded_and_conflicting_evidence_fail_closed() -> None:
    superseded = evaluate_pre_generation_policy(
        "What is the rule?",
        context(scope=JurisdictionScope(), source_status="superseded"),
    )
    conflicting = evaluate_pre_generation_policy(
        "What is the rule?",
        context(scope=JurisdictionScope(), conflicting_evidence=True),
    )

    assert superseded.reason is AbstentionReason.SUPERSEDED_SOURCE
    assert conflicting.reason is AbstentionReason.CONFLICTING_EVIDENCE


def test_deferred_semantic_support_never_claims_nli_support() -> None:
    assessment = DeferredSemanticSupport().assess("claim", "evidence")

    assert assessment.available is False
    assert assessment.supported is None
