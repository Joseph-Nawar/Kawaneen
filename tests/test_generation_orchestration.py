from __future__ import annotations

import json
from pathlib import Path

from kawaneen.generation.contracts import (
    AbstentionReason,
    GenerationDecision,
    GenerationResult,
    TokenizerFingerprint,
)
from kawaneen.generation.ollama import (
    OllamaModelIdentity,
    load_local_model_lock,
    write_local_model_lock,
)
from kawaneen.generation.orchestration import (
    STAGE_B_CHECKPOINT_ROOT,
    STAGE_C_CHECKPOINT_ROOT,
    STAGE_C_GENERATOR_NAME,
    RuntimeQuery,
    generation_fingerprint,
    generation_status,
    load_runtime_dev_queries,
    run_dev_generation,
)
from kawaneen.generation.tokenizer import CodepointTokenizer
from kawaneen.grounding.contracts import ContextPack, EvidenceReference, SourceRecord
from kawaneen.grounding.provenance import CanonicalCorpusResolver


def pack() -> ContextPack:
    return ContextPack(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
        assembly_policy_version="phase9-v1",
        token_counter_identity="phase9-v1",
        max_context_tokens=9999,
        token_count=0,
        units=(),
        blocks=(),
        evidence=(
            EvidenceReference(
                evidence_id="E001",
                unit_id="u1",
                block_id="B001",
                document_id="doc-1",
                display_text="The rule.",
                source=SourceRecord(document_id="doc-1"),
                contributing_chunk_ids=("chunk-1",),
                contributing_ranks=(1,),
            ),
        ),
        omissions=(),
        input_chunk_ids=("chunk-1",),
        chunk_policy_hash="c" * 64,
    )


class FailingQwen:
    def __init__(self) -> None:
        self.calls = 0
        self.last_raw_response = '{"private_source_text":"must remain private"}'

    def generate(self, _request: object) -> GenerationResult:
        self.calls += 1
        return GenerationResult(
            decision=GenerationDecision.ABSTAIN,
            abstention_reason=AbstentionReason.INVALID_GENERATION,
        )


class CapturingQwen:
    def __init__(self) -> None:
        self.calls = 0
        self.context_pack = None

    def generate(self, request: object) -> GenerationResult:
        self.calls += 1
        self.context_pack = request.context_pack  # type: ignore[attr-defined]
        return GenerationResult(
            decision=GenerationDecision.ABSTAIN,
            abstention_reason=AbstentionReason.INVALID_GENERATION,
        )


def test_status_reads_only_checkpoint_metadata(tmp_path: Path) -> None:
    result = generation_status("qwen-ollama", checkpoint_root=tmp_path / "checkpoints")

    assert result["generator"] == "qwen-ollama"
    assert result["completed"] == 0
    assert result["corrupt"] == 0


def test_stage_c_status_is_isolated_and_does_not_load_model_or_source(tmp_path: Path) -> None:
    result = generation_status(STAGE_C_GENERATOR_NAME, checkpoint_root=tmp_path / "checkpoints")

    assert result["generator"] == STAGE_C_GENERATOR_NAME
    assert result["model_loaded"] is False
    assert result["source_loaded"] is False
    assert STAGE_C_CHECKPOINT_ROOT != STAGE_B_CHECKPOINT_ROOT


def test_stage_c_status_separates_ready_contexts_from_legacy_checkpoints(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "q1.json").write_text(
        json.dumps(
            {
                "query_id": "q1",
                "generator_name": STAGE_C_GENERATOR_NAME,
                "result_path": "results/q1.json",
                "fingerprint": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    context_root = tmp_path / "context_packs"
    registry_root = tmp_path / "quote_registries"
    context_root.mkdir()
    registry_root.mkdir()
    (context_root / "q1.json").write_text("{}", encoding="utf-8")
    (registry_root / "q1.json").write_text("{}", encoding="utf-8")

    result = generation_status(
        STAGE_C_GENERATOR_NAME,
        checkpoint_root=checkpoint_root,
        context_cache_root=context_root,
        registry_root=registry_root,
    )

    assert result["expected"] == 160
    assert result["generation_completed"] == 0
    assert result["generation_missing"] == 159
    assert result["contexts_ready"] == 1
    assert result["quote_registries_ready"] == 1
    assert result["incomplete"] == 1
    assert result["corrupt"] == 0


def test_status_accepts_canonical_local_ollama_lock(tmp_path: Path) -> None:
    lock = tmp_path / "ollama-lock.json"
    write_local_model_lock(
        lock,
        OllamaModelIdentity(
            model="qwen3:4b-instruct-2507-q4_K_M",
            digest="sha256:" + "a" * 64,
        ),
    )

    assert load_local_model_lock(lock).digest == "sha256:" + "a" * 64
    assert (
        generation_status("qwen-ollama", checkpoint_root=tmp_path / "checkpoints")["model_loaded"]
        is False
    )


def test_runtime_query_loader_does_not_expose_qrels(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query_text": "What is the rule?",
                "split": "dev",
                "chunk_qrels": [{"chunk_id": "secret", "grade": 2}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    queries = load_runtime_dev_queries(path)

    assert queries == (RuntimeQuery(query_id="q1", query="What is the rule?"),)
    assert not hasattr(queries[0], "chunk_qrels")


def test_runtime_query_loader_excludes_holdout_records(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "query_id": query_id,
                    "query_text": "What is the rule?",
                    "split": split,
                }
            )
            for query_id, split in (("q-dev", "dev"), ("q-holdout", "holdout"))
        )
        + "\n",
        encoding="utf-8",
    )

    queries = load_runtime_dev_queries(path)

    assert tuple(item.query_id for item in queries) == ("q-dev",)


def test_qwen_failure_abstains_without_extractive_fallback(tmp_path: Path) -> None:
    qwen = FailingQwen()

    result = run_dev_generation(
        generator_name="qwen-ollama",
        resume=False,
        generator=qwen,
        tokenizer=CodepointTokenizer(),
        runtime_queries=(RuntimeQuery(query_id="q1", query="What is the rule?"),),
        context_packs={"q1": pack()},
        checkpoint_root=tmp_path / "checkpoints",
        results_root=tmp_path / "results",
        model_revision="d" * 40,
        ollama_digest="sha256:" + "e" * 64,
    )

    assert result["abstentions"] == 1
    assert qwen.calls == 1
    saved = json.loads((tmp_path / "results" / "q1.json").read_text(encoding="utf-8"))
    assert saved["result"]["abstention_reason"] == "INVALID_GENERATION"
    assert saved["raw_output"] == qwen.last_raw_response


def test_stage_c_resume_uses_only_stage_c_namespace_and_no_fallback(tmp_path: Path) -> None:
    kwargs = {
        "generator_name": STAGE_C_GENERATOR_NAME,
        "resume": False,
        "tokenizer": CodepointTokenizer(),
        "runtime_queries": (RuntimeQuery(query_id="q1", query="What is the rule?"),),
        "context_packs": {"q1": pack()},
        "assembler_factories": {
            "q1": lambda budget: pack().model_copy(update={"max_context_tokens": budget})
        },
        "checkpoint_root": tmp_path / "checkpoints",
        "results_root": tmp_path / "results",
        "context_cache_root": tmp_path / "context_packs",
        "model_revision": "d" * 40,
        "ollama_digest": "sha256:" + "e" * 64,
        "resolver": CanonicalCorpusResolver({}, {}, {}),
    }
    first = FailingQwen()
    first_result = run_dev_generation(generator=first, **kwargs)
    second = FailingQwen()
    kwargs["resume"] = True
    second_result = run_dev_generation(generator=second, **kwargs)

    assert first_result["abstentions"] == 1
    assert second_result["resumed"] == 1
    assert second.calls == 0
    checkpoint = json.loads((tmp_path / "checkpoints" / "q1.json").read_text())
    assert checkpoint["generator_name"] == STAGE_C_GENERATOR_NAME


def test_run_dev_builds_generator_context_without_phase9_private_pack(tmp_path: Path) -> None:
    seed = pack().model_copy(
        update={
            "evidence": (),
            "units": (),
            "blocks": (),
            "token_counter_identity": "codepoint-v1",
            "max_context_tokens": 0,
            "token_count": 0,
        }
    )
    generator = CapturingQwen()

    result = run_dev_generation(
        generator_name="qwen-ollama",
        resume=False,
        generator=generator,
        tokenizer=CodepointTokenizer(),
        runtime_queries=(RuntimeQuery(query_id="q1", query="What is the rule?"),),
        context_packs={"q1": seed},
        assembler_factories={
            "q1": lambda budget: pack().model_copy(
                update={"token_counter_identity": "codepoint-v1", "max_context_tokens": budget}
            )
        },
        checkpoint_root=tmp_path / "checkpoints",
        results_root=tmp_path / "results",
        context_cache_root=tmp_path / "context_packs",
        model_revision="d" * 40,
        ollama_digest="sha256:" + "e" * 64,
    )

    assert result["total"] == 1
    assert generator.calls == 1
    assert generator.context_pack is not None
    assert generator.context_pack.evidence  # type: ignore[union-attr]
    assert (tmp_path / "context_packs" / "q1.json").is_file()


def test_run_dev_resume_reuses_generator_context_and_result(tmp_path: Path) -> None:
    seed = pack().model_copy(
        update={
            "evidence": (),
            "units": (),
            "blocks": (),
            "token_counter_identity": "codepoint-v1",
            "max_context_tokens": 0,
            "token_count": 0,
        }
    )
    kwargs = {
        "generator_name": "qwen-ollama",
        "resume": False,
        "tokenizer": CodepointTokenizer(),
        "runtime_queries": (RuntimeQuery(query_id="q1", query="What is the rule?"),),
        "context_packs": {"q1": seed},
        "assembler_factories": {
            "q1": lambda budget: pack().model_copy(
                update={"token_counter_identity": "codepoint-v1", "max_context_tokens": budget}
            )
        },
        "checkpoint_root": tmp_path / "checkpoints",
        "results_root": tmp_path / "results",
        "context_cache_root": tmp_path / "context_packs",
        "model_revision": "d" * 40,
        "ollama_digest": "sha256:" + "e" * 64,
    }
    first = CapturingQwen()
    run_dev_generation(generator=first, **kwargs)
    second = CapturingQwen()
    kwargs["resume"] = True
    run = run_dev_generation(generator=second, **kwargs)

    assert run["resumed"] == 1
    assert second.calls == 0


def test_resume_skips_a_valid_completed_query(tmp_path: Path) -> None:
    first = FailingQwen()
    kwargs = {
        "generator_name": "qwen-ollama",
        "resume": False,
        "generator": first,
        "tokenizer": CodepointTokenizer(),
        "runtime_queries": (RuntimeQuery(query_id="q1", query="What is the rule?"),),
        "context_packs": {"q1": pack()},
        "checkpoint_root": tmp_path / "checkpoints",
        "results_root": tmp_path / "results",
        "model_revision": "d" * 40,
        "ollama_digest": "sha256:" + "e" * 64,
    }
    run_dev_generation(**kwargs)
    second = FailingQwen()
    kwargs["resume"] = True
    kwargs["generator"] = second

    result = run_dev_generation(**kwargs)

    assert result["resumed"] == 1
    assert second.calls == 0


def test_resume_recomputes_a_corrupt_checkpoint(tmp_path: Path) -> None:
    first = FailingQwen()
    kwargs = {
        "generator_name": "qwen-ollama",
        "resume": False,
        "generator": first,
        "tokenizer": CodepointTokenizer(),
        "runtime_queries": (RuntimeQuery(query_id="q1", query="What is the rule?"),),
        "context_packs": {"q1": pack()},
        "checkpoint_root": tmp_path / "checkpoints",
        "results_root": tmp_path / "results",
        "model_revision": "d" * 40,
        "ollama_digest": "sha256:" + "e" * 64,
    }
    run_dev_generation(**kwargs)
    (tmp_path / "checkpoints" / "q1.json").write_text("not-json", encoding="utf-8")
    second = FailingQwen()
    kwargs["resume"] = True
    kwargs["generator"] = second

    result = run_dev_generation(**kwargs)

    assert result["resumed"] == 0
    assert second.calls == 1


def test_generation_fingerprint_changes_with_model_tokenizer_or_prompt() -> None:
    tokenizer = CodepointTokenizer()
    base = generation_fingerprint(
        query_id="q1",
        context_pack=pack(),
        model_revision="a" * 40,
        ollama_digest="sha256:" + "b" * 64,
        tokenizer_fingerprint=tokenizer.fingerprint,
        prompt_template_hash="c" * 64,
        generation_policy_hash="d" * 64,
    )

    changed = generation_fingerprint(
        query_id="q1",
        context_pack=pack(),
        model_revision="e" * 40,
        ollama_digest="sha256:" + "b" * 64,
        tokenizer_fingerprint=TokenizerFingerprint(identity="qwen", revision="f" * 40),
        prompt_template_hash="1" * 64,
        generation_policy_hash="d" * 64,
    )

    assert base != changed
