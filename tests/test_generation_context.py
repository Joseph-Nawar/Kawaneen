from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawaneen.generation.context import (
    assemble_or_load_generator_context,
    generator_context_fingerprint,
)
from kawaneen.generation.contracts import GenerationSettings, TokenizerFingerprint
from kawaneen.generation.tokenizer import CodepointTokenizer
from kawaneen.grounding.contracts import ContextPack


def seed_pack(*, tokenizer_identity: str = "seed-v1") -> ContextPack:
    return ContextPack(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
        assembly_policy_version="phase9-context-assembly-v1",
        token_counter_identity=tokenizer_identity,
        max_context_tokens=4096,
        token_count=0,
        units=(),
        blocks=(),
        evidence=(),
        omissions=(),
        input_chunk_ids=("chunk-1", "chunk-2"),
        chunk_policy_hash="c" * 64,
    )


class LockedFakeQwenTokenizer:
    @property
    def fingerprint(self) -> TokenizerFingerprint:
        return TokenizerFingerprint(identity="Qwen/Qwen3-4B-Instruct-2507", revision="d" * 40)

    def count(self, text: str) -> int:
        return len(text)


def test_generator_context_builds_without_phase9_private_pack(tmp_path: Path) -> None:
    calls: list[int] = []
    tokenizer = CodepointTokenizer()

    def assemble(budget: int) -> ContextPack:
        calls.append(budget)
        return seed_pack(tokenizer_identity="codepoint-v1").model_copy(
            update={"max_context_tokens": budget}
        )

    result = assemble_or_load_generator_context(
        query="What is the rule?",
        context_seed=seed_pack(),
        tokenizer=tokenizer,
        assembler_factory=assemble,
        settings=GenerationSettings(total_input_tokens=3584),
        phase9_policy_hash="d" * 64,
        cache_root=tmp_path / "context_packs",
    )

    assert calls
    assert result.context_pack.token_counter_identity == "codepoint-v1"
    assert not (Path("artifacts/private/phase9_grounding") / "q1.json").exists()
    assert (tmp_path / "context_packs" / "q1.json").is_file()


def test_runtime_assembly_uses_ordered_phase8_top8_inputs(tmp_path: Path) -> None:
    tokenizer = CodepointTokenizer()
    captured: list[int] = []

    def assemble(budget: int) -> ContextPack:
        captured.append(budget)
        return seed_pack(tokenizer_identity="codepoint-v1").model_copy(
            update={"max_context_tokens": budget}
        )

    result = assemble_or_load_generator_context(
        query="What is the rule?",
        context_seed=seed_pack(),
        tokenizer=tokenizer,
        assembler_factory=assemble,
        settings=GenerationSettings(),
        phase9_policy_hash="d" * 64,
        cache_root=tmp_path / "context_packs",
    )

    assert captured and result.context_pack.input_chunk_ids == ("chunk-1", "chunk-2")


def test_context_fingerprint_includes_qwen_tokenizer_identity_and_budget() -> None:
    base = generator_context_fingerprint(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        input_chunk_ids=("chunk-1", "chunk-2"),
        phase9_policy_hash="b" * 64,
        tokenizer_fingerprint=TokenizerFingerprint(identity="Qwen", revision="c" * 40),
        settings=GenerationSettings(),
        evidence_token_budget=100,
    )
    changed_tokenizer = generator_context_fingerprint(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        input_chunk_ids=("chunk-1", "chunk-2"),
        phase9_policy_hash="b" * 64,
        tokenizer_fingerprint=TokenizerFingerprint(identity="Qwen", revision="d" * 40),
        settings=GenerationSettings(),
        evidence_token_budget=100,
    )
    changed_budget = generator_context_fingerprint(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        input_chunk_ids=("chunk-1", "chunk-2"),
        phase9_policy_hash="b" * 64,
        tokenizer_fingerprint=TokenizerFingerprint(identity="Qwen", revision="c" * 40),
        settings=GenerationSettings(total_input_tokens=3000),
        evidence_token_budget=100,
    )

    assert base != changed_tokenizer
    assert base != changed_budget


def test_codepoint_phase9_seed_is_not_consumed_as_qwen_context(tmp_path: Path) -> None:
    tokenizer = LockedFakeQwenTokenizer()
    seed = seed_pack(tokenizer_identity="codepoint-v1")

    result = assemble_or_load_generator_context(
        query="What is the rule?",
        context_seed=seed,
        tokenizer=tokenizer,
        assembler_factory=lambda budget: seed.model_copy(
            update={
                "token_counter_identity": (
                    "Qwen/Qwen3-4B-Instruct-2507:" + "d" * 40
                ),
                "max_context_tokens": budget,
            }
        ),
        settings=GenerationSettings(),
        phase9_policy_hash="b" * 64,
        cache_root=tmp_path / "context_packs",
    )

    assert result.context_pack.token_counter_identity.startswith("Qwen/")


def test_changed_tokenizer_or_budget_rebuilds_cached_context(tmp_path: Path) -> None:
    calls = 0

    def assemble(budget: int) -> ContextPack:
        nonlocal calls
        calls += 1
        return seed_pack(tokenizer_identity="codepoint-v1").model_copy(
            update={"max_context_tokens": budget}
        )

    kwargs = {
        "query": "What is the rule?",
        "context_seed": seed_pack(),
        "assembler_factory": assemble,
        "phase9_policy_hash": "d" * 64,
        "cache_root": tmp_path / "context_packs",
    }
    assemble_or_load_generator_context(tokenizer=CodepointTokenizer(), **kwargs)
    assemble_or_load_generator_context(tokenizer=CodepointTokenizer(), **kwargs)
    assemble_or_load_generator_context(
        tokenizer=CodepointTokenizer(),
        settings=GenerationSettings(total_input_tokens=3000),
        **{key: value for key, value in kwargs.items() if key != "settings"},
    )

    assert calls == 2


def test_corrupt_generator_context_is_rebuilt(tmp_path: Path) -> None:
    calls = 0

    def assemble(budget: int) -> ContextPack:
        nonlocal calls
        calls += 1
        return seed_pack(tokenizer_identity="codepoint-v1").model_copy(
            update={"max_context_tokens": budget}
        )

    kwargs = {
        "query": "What is the rule?",
        "context_seed": seed_pack(),
        "tokenizer": CodepointTokenizer(),
        "assembler_factory": assemble,
        "settings": GenerationSettings(),
        "phase9_policy_hash": "d" * 64,
        "cache_root": tmp_path / "context_packs",
    }
    assemble_or_load_generator_context(**kwargs)
    (tmp_path / "context_packs" / "q1.json").write_text("not-json", encoding="utf-8")
    assemble_or_load_generator_context(**kwargs)

    assert calls == 2


def test_cache_metadata_is_text_free_about_qrels(tmp_path: Path) -> None:
    assemble_or_load_generator_context(
        query="What is the rule?",
        context_seed=seed_pack(),
        tokenizer=CodepointTokenizer(),
        assembler_factory=lambda budget: seed_pack(tokenizer_identity="codepoint-v1").model_copy(
            update={"max_context_tokens": budget}
        ),
        settings=GenerationSettings(),
        phase9_policy_hash="d" * 64,
        cache_root=tmp_path / "context_packs",
    )
    payload = json.loads((tmp_path / "context_packs" / "q1.json").read_text())

    assert "qrels" not in json.dumps(payload).casefold()


def test_missing_provenance_input_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing provenance"):
        assemble_or_load_generator_context(
            query="What is the rule?",
            context_seed=seed_pack(),
            tokenizer=CodepointTokenizer(),
            assembler_factory=lambda _budget: (_ for _ in ()).throw(
                ValueError("missing provenance for chunk-1")
            ),
            settings=GenerationSettings(),
            phase9_policy_hash="d" * 64,
            cache_root=Path("/tmp/kawaneen-context-test-missing"),
        )
