from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from kawaneen.generation.budgeting import budget_context
from kawaneen.generation.contracts import GenerationSettings, TokenizerFingerprint
from kawaneen.generation.tokenizer import CodepointTokenizer, LazyHuggingFaceTokenizer
from kawaneen.grounding.contracts import ContextPack, EvidenceReference, SourceRecord


def pack(*rows: tuple[str, str]) -> ContextPack:
    source = SourceRecord(document_id="doc-1", document_title="Law")
    evidence = tuple(
        EvidenceReference(
            evidence_id=evidence_id,
            unit_id=unit_id,
            block_id="B001",
            document_id="doc-1",
            display_text=text,
            source=source,
            contributing_chunk_ids=(f"chunk-{unit_id}",),
            contributing_ranks=(1,),
        )
        for evidence_id, unit_id, text in rows
    )
    return ContextPack(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
        assembly_policy_version="phase9-v1",
        token_counter_identity="phase9-codepoint-v1",
        max_context_tokens=9999,
        token_count=0,
        units=(),
        blocks=(),
        evidence=evidence,
        omissions=(),
    )


def test_codepoint_tokenizer_has_stable_fingerprint() -> None:
    tokenizer = CodepointTokenizer()

    assert tokenizer.fingerprint == TokenizerFingerprint(identity="codepoint-v1")
    assert tokenizer.count("abc") == 3


def test_budget_counts_non_evidence_overhead_before_calling_phase9_assembler() -> None:
    original = pack(("E001", "u1", "The deadline is thirty days."))
    calls: list[int] = []

    def assembler_factory(max_tokens: int) -> ContextPack:
        calls.append(max_tokens)
        return original

    result = budget_context(
        query="deadline",
        context_pack=original,
        tokenizer=CodepointTokenizer(),
        assembler_factory=assembler_factory,
        settings=GenerationSettings(total_input_tokens=2000, safety_margin=5),
        gold_unit_ids=("u1", "u2"),
    )

    assert calls
    assert calls[0] == 2000 - result.non_evidence_prompt_tokens - 5
    assert result.evidence_token_count > 0
    assert result.gold_evidence_retention == 0.5
    assert result.complete_gold_evidence_retention == 0.0
    assert result.prompt_token_count <= 2000


def test_budget_rejects_a_rendered_prompt_that_exceeds_the_contract() -> None:
    original = pack(("E001", "u1", "A very long source unit."))

    def assembler_factory(_max_tokens: int) -> ContextPack:
        return original

    try:
        budget_context(
            query="deadline",
            context_pack=original,
            tokenizer=CodepointTokenizer(),
            assembler_factory=assembler_factory,
            settings=GenerationSettings(total_input_tokens=10, safety_margin=0),
        )
    except ValueError as error:
        assert "budget" in str(error)
    else:
        raise AssertionError("expected budget failure")


def test_budget_prompt_injects_server_controlled_jurisdiction() -> None:
    original = pack(("E001", "u1", "The rule applies."))

    class CapturingTokenizer(CodepointTokenizer):
        def __init__(self) -> None:
            self.seen: list[str] = []

        def count(self, text: str) -> int:
            self.seen.append(text)
            return super().count(text)

    tokenizer = CapturingTokenizer()

    budget_context(
        query="What is the rule?",
        context_pack=original,
        tokenizer=tokenizer,
        settings=GenerationSettings(total_input_tokens=2000, safety_margin=5),
        jurisdiction_text="SA",
    )

    assert any("Server jurisdiction scope: SA" in text for text in tokenizer.seen)


@dataclass
class FakeHFTokenizer:
    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class FakeBatchEncoding(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        if key != "input_ids":
            raise KeyError(key)
        return [1, 2]

    def __iter__(self) -> Iterator[str]:
        return iter(("input_ids",))

    def __len__(self) -> int:
        return 1


def test_lazy_huggingface_tokenizer_accepts_transformers_batch_encoding() -> None:
    tokenizer = LazyHuggingFaceTokenizer(
        identity="Qwen/Qwen3-4B-Instruct-2507",
        revision="a" * 40,
        loader=lambda _identity, _revision: lambda _text: FakeBatchEncoding(),
    )

    assert tokenizer.count("one two") == 2


def test_lazy_huggingface_tokenizer_uses_injected_loader_without_importing_transformers() -> None:
    calls: list[tuple[str, str]] = []

    def loader(identity: str, revision: str) -> FakeHFTokenizer:
        calls.append((identity, revision))
        return FakeHFTokenizer()

    tokenizer = LazyHuggingFaceTokenizer(
        identity="Qwen/Qwen3-4B-Instruct-2507",
        revision="a" * 40,
        loader=loader,
    )
    assert calls == []
    assert tokenizer.count("one two") == 2
    assert calls == [("Qwen/Qwen3-4B-Instruct-2507", "a" * 40)]
