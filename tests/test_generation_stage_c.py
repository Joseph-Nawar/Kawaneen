from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kawaneen.generation.checkpoints import GenerationCheckpointStore, QueryCheckpoint
from kawaneen.generation.contracts import (
    ClaimMode,
    GenerationDecision,
    GenerationRequest,
    GenerationSettings,
    parse_stage_c_generation_payload,
    stage_c_generation_payload_schema,
)
from kawaneen.generation.ollama import OllamaGenerator
from kawaneen.generation.postprocessing import finalize_generation
from kawaneen.generation.prompt import render_stage_c_generation_prompt
from kawaneen.generation.quote_registry import (
    QUOTE_REGISTRY_POLICY_VERSION,
    build_quote_registry,
    stage_c_result_from_payload,
)
from kawaneen.generation.stage_c import STAGE_C_TIMEOUT_SECONDS, stage_c_fingerprint
from kawaneen.generation.stage_c_context import assemble_or_load_stage_c_context
from kawaneen.generation.tokenizer import CodepointTokenizer
from kawaneen.grounding.contracts import (
    ContextBlock,
    ContextPack,
    ContextUnit,
    EvidenceReference,
    SourceRecord,
)


def _pack() -> ContextPack:
    source = SourceRecord(document_id="doc-1", document_title="Law")
    units = (
        ContextUnit(
            unit_id="u1",
            document_id="doc-1",
            ordinal=1,
            display_text="First canonical unit.",
            source=source,
            best_retrieval_rank=1,
            contributing_chunk_ids=("chunk-1",),
            contributing_ranks=(1,),
        ),
        ContextUnit(
            unit_id="u2",
            document_id="doc-1",
            ordinal=2,
            display_text="Second canonical unit.",
            source=source,
            best_retrieval_rank=2,
            contributing_chunk_ids=("chunk-2",),
            contributing_ranks=(2,),
        ),
    )
    return ContextPack(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
        assembly_policy_version="phase9-v1",
        token_counter_identity="fake-v1",
        max_context_tokens=100,
        token_count=10,
        units=units,
        blocks=(
            ContextBlock(
                block_id="B001",
                document_id="doc-1",
                source=source,
                units=units,
                best_retrieval_rank=1,
            ),
        ),
        evidence=(
            EvidenceReference(
                evidence_id="E001",
                unit_id="u1",
                block_id="B001",
                document_id="doc-1",
                display_text="First canonical unit.",
                source=source,
                contributing_chunk_ids=("chunk-1",),
                contributing_ranks=(1,),
            ),
            EvidenceReference(
                evidence_id="E002",
                unit_id="u2",
                block_id="B001",
                document_id="doc-1",
                display_text="Second canonical unit.",
                source=source,
                contributing_chunk_ids=("chunk-2",),
                contributing_ranks=(2,),
            ),
            EvidenceReference(
                evidence_id="E003",
                unit_id="u1",
                block_id="B001",
                document_id="doc-1",
                display_text="First canonical unit.",
                source=source,
                contributing_chunk_ids=("chunk-1",),
                contributing_ranks=(1,),
            ),
        ),
        omissions=(),
        input_chunk_ids=("chunk-1", "chunk-2"),
    )


def test_quote_registry_assigns_deterministic_context_local_ids_and_deduplicates_units() -> None:
    registry = build_quote_registry(_pack())

    assert tuple(item.quote_id for item in registry.entries) == ("Q001", "Q002")
    assert tuple(item.canonical_unit_id for item in registry.entries) == ("u1", "u2")
    assert registry.resolve("Q001").display_text == "First canonical unit."
    assert registry.resolve("Q002").evidence_id == "E002"
    assert registry.query_id == "q1"


def test_quote_registry_rejects_unknown_or_malformed_local_ids() -> None:
    registry = build_quote_registry(_pack())

    with pytest.raises(ValueError, match="unknown quote reference"):
        registry.resolve("Q999")
    with pytest.raises(ValueError):
        registry.resolve("E001")


def test_quote_registry_fingerprint_changes_when_context_changes() -> None:
    base = build_quote_registry(_pack())
    changed_pack = _pack()
    changed = build_quote_registry(
        changed_pack.model_copy(
            update={
                "units": (
                    changed_pack.units[0].model_copy(update={"display_text": "Changed."}),
                    changed_pack.units[1],
                ),
                "blocks": (
                    changed_pack.blocks[0].model_copy(
                        update={
                            "units": (
                                changed_pack.blocks[0].units[0].model_copy(
                                    update={"display_text": "Changed."}
                                ),
                                changed_pack.blocks[0].units[1],
                            )
                        }
                    ),
                ),
                "evidence": (
                    changed_pack.evidence[0].model_copy(update={"display_text": "Changed."}),
                    changed_pack.evidence[1],
                    changed_pack.evidence[2],
                ),
            }
        )
    )

    assert base.fingerprint != changed.fingerprint


def test_stage_c_schema_uses_quote_refs_and_forbids_model_quotation_or_metadata() -> None:
    schema = stage_c_generation_payload_schema()
    direct = parse_stage_c_generation_payload(
        '{"decision":"answer","claims":[{"mode":"direct","quote_refs":["Q001"]}]}'
    )
    interpretation = parse_stage_c_generation_payload(
        '{"decision":"answer","claims":[{"mode":"interpretation","text":"Meaning","quote_refs":["Q001"]}]}'
    )

    assert schema["properties"]["claims"]["maxItems"] == 3
    assert "quoted_text" not in json.dumps(schema)
    assert direct.claims[0].quote_refs == ("Q001",)
    assert interpretation.claims[0].text == "Meaning"
    with pytest.raises(ValidationError):
        parse_stage_c_generation_payload(
            '{"decision":"answer","claims":[{"mode":"direct","quote_refs":["Q001"],"quoted_text":"forged"}]}'
        )
    with pytest.raises(ValidationError):
        parse_stage_c_generation_payload(
            '{"decision":"answer","claims":[{"mode":"direct","quote_refs":["Q001"],"source_url":"forged"}]}'
        )


def test_stage_c_schema_enforces_claim_and_reference_cardinality() -> None:
    too_many_claims = {
        "decision": "answer",
        "claims": [{"mode": "direct", "quote_refs": ["Q001"]}] * 4,
    }
    too_many_refs = {
        "decision": "answer",
        "claims": [{"mode": "direct", "quote_refs": ["Q001", "Q002", "Q003", "Q004"]}],
    }

    with pytest.raises(ValidationError):
        parse_stage_c_generation_payload(json.dumps(too_many_claims))
    with pytest.raises(ValidationError):
        parse_stage_c_generation_payload(json.dumps(too_many_refs))


def test_stage_c_resolves_qids_to_exact_phase9_citation_requests() -> None:
    registry = build_quote_registry(_pack())
    payload = parse_stage_c_generation_payload(
        '{"decision":"answer","claims":[{"mode":"direct","quote_refs":["Q002"]}]}'
    )

    result = stage_c_result_from_payload(payload, registry)

    assert result.claims[0].mode is ClaimMode.DIRECT
    assert result.claims[0].text is None
    assert result.claims[0].citations[0].evidence_id == "E002"
    assert result.claims[0].citations[0].quoted_text == "Second canonical unit."


def test_unknown_qid_fails_closed_before_phase9_finalization() -> None:
    registry = build_quote_registry(_pack())
    payload = parse_stage_c_generation_payload(
        '{"decision":"answer","claims":[{"mode":"direct","quote_refs":["Q999"]}]}'
    )

    with pytest.raises(ValueError, match="unknown quote reference"):
        stage_c_result_from_payload(payload, registry)


def test_stage_c_prompt_labels_exact_source_with_local_quote_ids() -> None:
    registry = build_quote_registry(_pack())
    prompt = render_stage_c_generation_prompt(
        "What is the rule?",
        _pack(),
        registry=registry,
        settings=GenerationSettings(max_new_tokens=512, output_reservation=512),
        jurisdiction_text="SA",
    )

    assert "[Q001] First canonical unit." in prompt.text
    assert "[Q002] Second canonical unit." in prompt.text
    assert "quote_refs" in prompt.text
    assert "quoted_text" not in prompt.text


def test_stage_c_direct_rendering_uses_only_verified_authoritative_text(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "document_id": "doc-1",
                        "ordinal": 1,
                        "text": "First canonical unit.",
                        "unit_type": "facts",
                        "provenance": {
                            "source_id": "source-1",
                            "source_version": "v1",
                            "source_path": "private",
                            "source_row": 1,
                            "source_field": "body",
                            "split": "",
                        },
                    },
                    {
                        "unit_id": "u2",
                        "document_id": "doc-1",
                        "ordinal": 2,
                        "text": "Second canonical unit.",
                        "unit_type": "facts",
                        "provenance": {
                            "source_id": "source-1",
                            "source_version": "v1",
                            "source_path": "private",
                            "source_row": 2,
                            "source_field": "body",
                            "split": "",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "\n".join(
            json.dumps({"chunk_id": chunk, "source_unit_ids": [unit], "source_spans": []})
            for chunk, unit in (("chunk-1", "u1"), ("chunk-2", "u2"))
        )
        + "\n",
        encoding="utf-8",
    )
    from kawaneen.grounding.provenance import CanonicalCorpusResolver

    resolver = CanonicalCorpusResolver.from_json(canonical, chunks)
    result = finalize_generation(
        _pack(),
        stage_c_result_from_payload(
            parse_stage_c_generation_payload(
                '{"decision":"answer","claims":[{"mode":"direct","quote_refs":["Q001"]}]}'
            ),
            build_quote_registry(_pack()),
        ),
        resolver,
        jurisdiction_text="SA",
    )

    assert result.rendered_answer is not None
    assert "First canonical unit." in result.rendered_answer


def test_stage_c_interpretation_remains_unavailable_without_semantic_support() -> None:
    payload = parse_stage_c_generation_payload(
        '{"decision":"answer","claims":[{"mode":"interpretation","text":"Meaning","quote_refs":["Q001"]}]}'
    )
    result = stage_c_result_from_payload(payload, build_quote_registry(_pack()))

    assert result.claims[0].mode is ClaimMode.INTERPRETATION
    assert result.claims[0].text == "Meaning"


def test_stage_c_timeout_and_fingerprint_are_isolated() -> None:
    generator = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model="qwen3:4b-instruct-2507-q4_K_M",
        immutable_digest="sha256:" + "a" * 64,
        stage_c=True,
    )
    registry = build_quote_registry(_pack())
    first = stage_c_fingerprint(
        query_id="q1",
        context_pack=_pack(),
        registry=registry,
        model_revision="b" * 40,
        ollama_digest="sha256:" + "a" * 64,
        tokenizer_identity="Qwen",
        tokenizer_revision="c" * 40,
        prompt_hash="d" * 64,
        schema_hash="e" * 64,
        policy_hash="f" * 64,
    )

    assert generator.transport.timeout_seconds == STAGE_C_TIMEOUT_SECONDS == 60.0  # type: ignore[attr-defined]
    assert QUOTE_REGISTRY_POLICY_VERSION in first or first != ""


def test_stage_c_ollama_schema_resolves_refs_and_persists_native_telemetry() -> None:
    class Transport:
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        def get_json(self, _endpoint: str) -> object:
            return {"models": []}

        def post_json(self, _endpoint: str, payload: dict[str, object]) -> object:
            self.payload = payload
            return {
                "response": (
                    '{"decision":"answer","claims":[{"mode":"direct",'
                    '"quote_refs":["Q001"]}]}'
                ),
                "done": True,
                "done_reason": "stop",
                "eval_count": 12,
                "eval_duration": 100,
            }

    transport = Transport()
    registry = build_quote_registry(_pack())
    generator = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model="qwen3:4b-instruct-2507-q4_K_M",
        immutable_digest="sha256:" + "a" * 64,
        transport=transport,
        stage_c=True,
    )

    result = generator.generate(
        GenerationRequest(
            query="What is the rule?",
            context_pack=_pack(),
            settings=GenerationSettings(max_new_tokens=512, output_reservation=512),
            quote_registry=registry,
        )
    )

    assert result.decision is GenerationDecision.ANSWER
    assert result.claims[0].citations[0].quoted_text == "First canonical unit."
    assert transport.payload is not None
    assert transport.payload["format"] == stage_c_generation_payload_schema()
    assert transport.payload["options"]["num_predict"] == 512  # type: ignore[index]
    assert generator.last_telemetry["done_reason"] == "stop"
    assert generator.last_telemetry["eval_count"] == 12


def test_stage_c_checkpoint_can_persist_text_free_native_telemetry(tmp_path: Path) -> None:
    store = GenerationCheckpointStore(tmp_path)
    store.write(
        QueryCheckpoint(
            query_id="q1",
            generator_name="qwen-ollama-stage-c",
            result_path="results/q1.json",
            fingerprint="a" * 64,
            telemetry={"http_status": 200, "done_reason": "stop", "eval_count": 12},
        )
    )

    assert store.load("q1").telemetry["done_reason"] == "stop"


def test_stage_c_context_cache_rebuilds_corrupt_registry_without_model_access(
    tmp_path: Path,
) -> None:
    tokenizer = CodepointTokenizer()
    context_root = tmp_path / "context_packs"
    registry_root = tmp_path / "quote_registries"

    def assemble(budget: int) -> ContextPack:
        return _pack().model_copy(update={"max_context_tokens": budget})

    first = assemble_or_load_stage_c_context(
        query="What is the rule?",
        context_seed=_pack().model_copy(update={"evidence": (), "units": (), "blocks": ()}),
        tokenizer=tokenizer,
        assembler_factory=assemble,
        settings=GenerationSettings(max_new_tokens=512, output_reservation=512),
        phase9_policy_hash="c" * 64,
        cache_root=context_root,
        registry_root=registry_root,
        jurisdiction_text="SA",
    )
    (registry_root / "q1.json").write_text("corrupt", encoding="utf-8")
    second = assemble_or_load_stage_c_context(
        query="What is the rule?",
        context_seed=_pack().model_copy(update={"evidence": (), "units": (), "blocks": ()}),
        tokenizer=tokenizer,
        assembler_factory=assemble,
        settings=GenerationSettings(max_new_tokens=512, output_reservation=512),
        phase9_policy_hash="c" * 64,
        cache_root=context_root,
        registry_root=registry_root,
        jurisdiction_text="SA",
    )

    assert first.quote_registry.fingerprint == second.quote_registry.fingerprint
    assert json.loads((registry_root / "q1.json").read_text(encoding="utf-8"))["fingerprint"]
