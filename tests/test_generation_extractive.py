from __future__ import annotations

from kawaneen.generation.contracts import (
    AbstentionReason,
    GenerationDecision,
    GenerationRequest,
)
from kawaneen.generation.extractive import ExtractiveGenerator, lexical_terms
from kawaneen.grounding.contracts import ContextPack, EvidenceReference, SourceRecord


def pack(*rows: tuple[str, str, int]) -> ContextPack:
    source = SourceRecord(document_id="doc-1", document_title="Law")
    evidence = tuple(
        EvidenceReference(
            evidence_id=evidence_id,
            unit_id=f"unit-{evidence_id}",
            block_id="B001",
            document_id="doc-1",
            display_text=text,
            source=source,
            contributing_chunk_ids=(f"chunk-{rank}",),
            contributing_ranks=(rank,),
        )
        for evidence_id, text, rank in rows
    )
    return ContextPack(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
        assembly_policy_version="phase9-v1",
        token_counter_identity="fake-v1",
        max_context_tokens=100,
        token_count=5,
        units=(),
        blocks=(),
        evidence=evidence,
        omissions=(),
    )


def request(query: str, context_pack: ContextPack) -> GenerationRequest:
    return GenerationRequest(query=query, context_pack=context_pack)


def test_extractive_generator_selects_at_most_two_exact_source_units() -> None:
    context = pack(
        ("E001", "The payment deadline is thirty days.", 2),
        ("E002", "The payment deadline is fourteen days.", 1),
        ("E003", "The court has jurisdiction.", 3),
    )

    result = ExtractiveGenerator().generate(request("payment deadline", context))

    assert result.decision is GenerationDecision.ANSWER
    assert len(result.claims) == 2
    assert [claim.text for claim in result.claims] == [
        "The payment deadline is fourteen days.",
        "The payment deadline is thirty days.",
    ]
    assert all(claim.text == claim.citations[0].quoted_text for claim in result.claims)


def test_extractive_generator_is_benchmark_only() -> None:
    assert ExtractiveGenerator.benchmark_only is True


def test_extractive_ties_use_evidence_id_and_never_rewrite_text() -> None:
    text = "المهلة 30 يوماً، ولا يجوز تمديدها."
    result = ExtractiveGenerator().generate(
        request("المهلة", pack(("E002", text, 1), ("E001", text, 1)))
    )

    assert [claim.citations[0].evidence_id for claim in result.claims] == ["E001", "E002"]
    assert result.claims[0].text == text


def test_extractive_generator_abstains_without_context_or_positive_overlap() -> None:
    generator = ExtractiveGenerator()

    no_context = generator.generate(request("deadline", pack()))
    no_overlap = generator.generate(
        request("appeal", pack(("E001", "The payment deadline is thirty days.", 1)))
    )

    assert no_context.abstention_reason is AbstentionReason.NO_CONTEXT
    assert no_overlap.abstention_reason is AbstentionReason.REQUESTED_INFO_NOT_FOUND


def test_lexical_terms_are_deterministic_for_arabic_and_english() -> None:
    assert lexical_terms("Payment، المَهلة 30") == frozenset({"payment", "المَهلة", "30"})
