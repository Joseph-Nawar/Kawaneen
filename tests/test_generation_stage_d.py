from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kawaneen.generation.answerability import (
    SourceEligibility,
    evaluate_stage_d_policy,
)
from kawaneen.generation.contracts import (
    AbstentionReason,
    GenerationDecision,
    GenerationRequest,
    parse_stage_d_generation_payload,
    stage_d_generation_payload_schema,
)
from kawaneen.generation.ollama import OllamaGenerator
from kawaneen.generation.policy import JurisdictionScope, PolicyContext
from kawaneen.generation.prompt import render_stage_d_generation_prompt
from kawaneen.generation.quote_registry import build_quote_registry, stage_d_result_from_payload
from kawaneen.generation.stage_d import STAGE_D_GENERATOR_NAME
from kawaneen.grounding.contracts import (
    ContextBlock,
    ContextPack,
    ContextUnit,
    EvidenceReference,
    SourceRecord,
)


def _pack(*, source_id: str = "case-law") -> ContextPack:
    source = SourceRecord(source_id=source_id, document_id="doc-1", document_title="Decision")
    unit = ContextUnit(
        unit_id="u1",
        document_id="doc-1",
        ordinal=1,
        display_text="The court decided the matter.",
        source=source,
        best_retrieval_rank=1,
        contributing_chunk_ids=("chunk-1",),
        contributing_ranks=(1,),
    )
    return ContextPack(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
        assembly_policy_version="phase9-v1",
        token_counter_identity="fake-v1",
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
                unit_id="u1",
                block_id="B001",
                document_id="doc-1",
                display_text=unit.display_text,
                source=source,
                contributing_chunk_ids=("chunk-1",),
                contributing_ranks=(1,),
            ),
        ),
        omissions=(),
    )


def _policy_context(
    *, source_id: str = "case-law", source_status_available: bool = False
) -> PolicyContext:
    return PolicyContext(
        context_pack=_pack(source_id=source_id),
        scope=JurisdictionScope(
            active_jurisdiction="SA", allowed_jurisdictions=("SA",), mode="single"
        ),
        source_status_available=source_status_available,
    )


def _official_source() -> SourceEligibility:
    return SourceEligibility(
        source_id="official",
        source_type="official_statute",
        source_role="official_primary",
        authority_level="official",
        decision="approved",
        scope_terms=("statute", "regulation"),
    )


def test_future_law_request_abstains() -> None:
    result = evaluate_stage_d_policy(
        "What law will be issued in the future?",
        _policy_context(),
    )

    assert result.reason is AbstentionReason.FUTURE_LAW_UNKNOWABLE


def test_arabic_future_law_request_abstains() -> None:
    result = evaluate_stage_d_policy(
        "أي قانون سيصدر مستقبلاً ليحكم واقعة؟",
        _policy_context(),
    )

    assert result.reason is AbstentionReason.FUTURE_LAW_UNKNOWABLE


def test_already_enacted_historical_amendment_is_not_future_law() -> None:
    result = evaluate_stage_d_policy(
        "What did the amendment enacted in 2020 provide?",
        _policy_context(),
    )

    assert result.allowed


def test_change_since_date_requires_currentness_evidence() -> None:
    result = evaluate_stage_d_policy(
        "Did the rule change since the judgment date?",
        _policy_context(source_status_available=False),
    )

    assert result.reason is AbstentionReason.CURRENTNESS_UNVERIFIED


def test_official_text_request_rejects_case_law_only_context() -> None:
    result = evaluate_stage_d_policy(
        "What is the official updated statutory text?",
        _policy_context(source_id="case-law"),
        source_registry={
            "case-law": SourceEligibility(
                source_id="case-law",
                source_type="dataset",
                source_role="benchmark",
                authority_level="academic",
                decision="evaluation_only",
            )
        },
    )

    assert result.reason is AbstentionReason.AUTHORITATIVE_SOURCE_UNAVAILABLE


def test_primary_authoritative_text_satisfies_source_requirement() -> None:
    result = evaluate_stage_d_policy(
        "What is the official statutory text?",
        _policy_context(source_id="official"),
        source_registry={"official": _official_source()},
    )

    assert result.allowed


def test_precedent_facts_cannot_establish_unspecified_user_facts() -> None:
    result = evaluate_stage_d_policy(
        "Was delivery established in the matter?",
        _policy_context(),
    )

    assert result.reason is AbstentionReason.CASE_FACTS_NOT_ESTABLISHED


def test_identified_case_may_support_its_own_facts() -> None:
    result = evaluate_stage_d_policy(
        "In judgment 123, was delivery established?",
        _policy_context(),
    )

    assert result.allowed


def test_missing_dispositive_section_abstains() -> None:
    result = evaluate_stage_d_policy(
        "What was the dispositive outcome?",
        _policy_context(),
        structural_roles={"u1": "reasoning"},
    )

    assert result.reason is AbstentionReason.REQUIRED_CASE_SECTION_MISSING


def test_query_identifying_absent_operative_part_abstains() -> None:
    result = evaluate_stage_d_policy(
        "What result is binding if the operative part is absent from the available materials?",
        _policy_context(),
        structural_roles={"u1": "ruling"},
    )

    assert result.reason is AbstentionReason.REQUIRED_CASE_SECTION_MISSING


def test_available_operative_holding_passes_dispositive_request() -> None:
    result = evaluate_stage_d_policy(
        "What was the dispositive outcome?",
        _policy_context(),
        structural_roles={"u1": "ruling"},
    )

    assert result.allowed


def test_explicit_forum_request_fails_closed_without_scope_metadata() -> None:
    result = evaluate_stage_d_policy(
        "How does a labor court handle this claim?",
        _policy_context(),
    )

    assert result.reason is AbstentionReason.FORUM_OR_SOURCE_SCOPE_MISMATCH


def test_policy_has_no_qrel_or_evaluation_label_dependency() -> None:
    context = _policy_context()
    assert not hasattr(context, "chunk_qrels")
    result = evaluate_stage_d_policy("What rule did the court state?", context)
    assert result.allowed


def test_stage_d_schema_accepts_only_direct_claims() -> None:
    payload = parse_stage_d_generation_payload(
        json.dumps({"decision": "answer", "claims": [{"mode": "direct", "quote_refs": ["Q001"]}]})
    )

    assert payload.decision is GenerationDecision.ANSWER
    with pytest.raises(ValidationError):
        parse_stage_d_generation_payload(
            json.dumps(
                {
                    "decision": "answer",
                    "claims": [{"mode": "interpretation", "text": "x", "quote_refs": ["Q001"]}],
                }
            )
        )


def test_stage_d_schema_forbids_extra_fields_and_exposes_no_interpretation() -> None:
    schema = stage_d_generation_payload_schema()
    schema_text = json.dumps(schema, sort_keys=True)
    assert "interpretation" not in schema_text
    assert "quote_refs" in schema_text
    with pytest.raises(ValidationError):
        parse_stage_d_generation_payload(
            json.dumps(
                {
                    "decision": "answer",
                    "claims": [{"mode": "direct", "quote_refs": ["Q001"], "text": "forbidden"}],
                }
            )
        )


def test_stage_d_schema_enforces_direct_claim_and_reference_limits() -> None:
    with pytest.raises(ValidationError):
        parse_stage_d_generation_payload(
            json.dumps(
                {
                    "decision": "answer",
                    "claims": [{"mode": "direct", "quote_refs": ["Q001", "Q002", "Q003", "Q004"]}],
                }
            )
        )
    with pytest.raises(ValidationError):
        parse_stage_d_generation_payload(
            json.dumps(
                {
                    "decision": "answer",
                    "claims": [{"mode": "direct", "quote_refs": ["Q001"]}] * 4,
                }
            )
        )


def test_stage_d_adapter_uses_direct_schema_and_frozen_runtime_settings() -> None:
    generator = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model="qwen3:4b-instruct-2507-q4_K_M",
        immutable_digest="sha256:" + "a" * 64,
        stage_d=True,
    )
    registry = build_quote_registry(_pack())
    payload = generator.build_payload(
        GenerationRequest(
            query="What rule did the court state?",
            context_pack=_pack(),
            quote_registry=registry,
        )
    )

    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.0, "num_predict": 512}
    assert isinstance(payload["format"], dict)
    schema_text = json.dumps(payload["format"], sort_keys=True)
    assert '"mode": "interpretation"' not in schema_text
    assert "quoted_text" not in schema_text
    assert generator.transport.timeout_seconds == 60.0


def test_stage_d_direct_payload_resolves_only_server_authoritative_text() -> None:
    registry = build_quote_registry(_pack())
    payload = parse_stage_d_generation_payload(
        '{"decision":"answer","claims":[{"mode":"direct","quote_refs":["Q001"]}]}'
    )
    result = stage_d_result_from_payload(payload, registry)

    assert result.claims[0].text is None
    assert result.claims[0].citations[0].quoted_text == "The court decided the matter."


def test_stage_d_prompt_is_versioned_and_direct_only() -> None:
    prompt = render_stage_d_generation_prompt(
        "What rule did the court state?",
        _pack(),
        registry=build_quote_registry(_pack()),
    )

    assert prompt.version_hash != "0" * 64
    assert "direct claims only" in prompt.text.lower()
    assert '"mode":"interpretation"' not in prompt.text.lower()
    assert STAGE_D_GENERATOR_NAME == "qwen-ollama-stage-d"
