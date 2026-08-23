from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kawaneen.generation.contracts import (
    STAGE_B_GENERATION_SETTINGS,
    ClaimMode,
    DirectClaim,
    GenerationDecision,
    GenerationPayload,
    InterpretationClaim,
    ModelOutput,
    ModelOutputCitation,
    ModelOutputClaim,
    VerifiedCitation,
    VerifiedClaim,
    generation_payload_schema,
    parse_generation_payload,
)
from kawaneen.generation.ollama import OllamaGenerator
from kawaneen.generation.orchestration import (
    STAGE_B_CHECKPOINT_ROOT,
    STAGE_B_GENERATOR_NAME,
    STAGE_B_RESULTS_ROOT,
    generation_status,
)
from kawaneen.generation.postprocessing import finalize_generation
from kawaneen.generation.rendering import render_verified_answer
from kawaneen.grounding.contracts import ContextPack, EvidenceReference, SourceRecord
from kawaneen.grounding.provenance import CanonicalCorpusResolver


def _pack() -> ContextPack:
    return ContextPack(
        query_id="q1",
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
        assembly_policy_version="phase9-v1",
        token_counter_identity="fake-v1",
        max_context_tokens=100,
        token_count=1,
        units=(),
        blocks=(),
        evidence=(
            EvidenceReference(
                evidence_id="E001",
                unit_id="u1",
                block_id="B001",
                document_id="doc-1",
                display_text="Exact authoritative wording.",
                source=SourceRecord(document_id="doc-1", document_title="Law"),
                contributing_chunk_ids=("c1",),
                contributing_ranks=(1,),
            ),
        ),
        omissions=(),
    )


def _resolver(tmp_path: Path) -> CanonicalCorpusResolver:
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "document_id": "doc-1",
                        "ordinal": 1,
                        "text": "Exact authoritative wording.",
                        "unit_type": "facts",
                        "provenance": {
                            "source_id": "source-1",
                            "source_version": "v1",
                            "source_path": "private",
                            "source_row": 1,
                            "source_field": "body",
                            "split": "",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps({"chunk_id": "c1", "source_unit_ids": ["u1"], "source_spans": []})
        + "\n",
        encoding="utf-8",
    )
    return CanonicalCorpusResolver.from_json(canonical, chunks)


def test_provider_schema_is_strict_and_matches_stage_b_contract() -> None:
    schema = generation_payload_schema()

    assert schema["properties"]["decision"]["enum"] == ["answer", "abstain"]
    assert schema["properties"]["claims"]["maxItems"] == 3
    assert schema["required"] == ["decision", "claims"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["DirectClaim"]["required"] == ["mode", "citations"]
    assert schema["$defs"]["InterpretationClaim"]["required"] == [
        "mode",
        "text",
        "citations",
    ]


def test_direct_and_interpretation_claims_have_explicit_modes() -> None:
    direct = parse_generation_payload(
        json.dumps(
            {
                "decision": "answer",
                "claims": [
                    {
                        "mode": "direct",
                        "citations": [{"evidence_id": "E001", "quoted_text": "Exact."}],
                    }
                ],
            }
        )
    )
    interpretation = parse_generation_payload(
        json.dumps(
            {
                "decision": "answer",
                "claims": [
                    {
                        "mode": "interpretation",
                        "text": "A supported interpretation.",
                        "citations": [{"evidence_id": "E001", "quoted_text": "Exact."}],
                    }
                ],
            }
        )
    )

    assert isinstance(direct.claims[0], DirectClaim)
    assert isinstance(interpretation.claims[0], InterpretationClaim)
    assert direct.claims[0].mode is ClaimMode.DIRECT
    assert interpretation.claims[0].mode is ClaimMode.INTERPRETATION


def test_stage_b_contract_rejects_extra_fields_and_more_than_three_claims() -> None:
    with pytest.raises(ValidationError):
        parse_generation_payload(
            json.dumps(
                {
                    "decision": "abstain",
                    "claims": [],
                    "unexpected": True,
                }
            )
        )
    claim = {"mode": "direct", "citations": [{"evidence_id": "E001", "quoted_text": "x"}]}
    with pytest.raises(ValidationError):
        GenerationPayload.model_validate({"decision": "answer", "claims": [claim] * 4})


def test_stage_b_direct_rendering_uses_verified_quotation_not_claim_text() -> None:
    citation = VerifiedCitation(
        evidence_id="E001",
        document_id="doc-1",
        document_title="Verified law",
        jurisdiction="SA",
        article=None,
        page=None,
        chunk_id="chunk-1",
        source_url=None,
        quoted_text="Exact authoritative wording.",
    )
    answer = render_verified_answer(
        (VerifiedClaim(mode=ClaimMode.DIRECT, text="ignored", citations=(citation,)),),
        jurisdiction_text="SA",
        disclaimer_text="",
    )

    assert "Exact authoritative wording." in answer
    assert "ignored" not in answer


def test_interpretation_without_semantic_support_abstains_distinctly(tmp_path: Path) -> None:
    result = finalize_generation(
        _pack(),
        ModelOutput(
            decision=GenerationDecision.ANSWER,
            claims=(
                ModelOutputClaim(
                    mode=ClaimMode.INTERPRETATION,
                    text="A paraphrase.",
                    citations=(
                        ModelOutputCitation(
                            evidence_id="E001",
                            quoted_text="Exact authoritative wording.",
                        ),
                    ),
                ),
            ),
        ),
        _resolver(tmp_path),
    )

    assert result.abstention_reason.value == "SEMANTIC_SUPPORT_UNAVAILABLE"


def test_ollama_sends_full_schema_and_stage_b_output_ceiling() -> None:
    class Transport:
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        def get_json(self, _endpoint: str) -> object:
            return {"models": []}

        def post_json(self, _endpoint: str, payload: dict[str, object]) -> object:
            self.payload = payload
            return {"response": '{"decision":"abstain","claims":[]}'}

    transport = Transport()
    generator = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model="qwen3:4b-instruct-2507-q4_K_M",
        immutable_digest="sha256:" + "a" * 64,
        transport=transport,
        stage_b=True,
    )
    from kawaneen.generation.contracts import GenerationRequest

    generator.generate(GenerationRequest(query="q", context_pack=_pack()))

    assert transport.payload["format"] == generation_payload_schema()  # type: ignore[index]
    assert transport.payload["options"]["num_predict"] == 512  # type: ignore[index]
    assert transport.payload["format"]["properties"]  # type: ignore[index]
    assert STAGE_B_GENERATION_SETTINGS.total_input_tokens == 3584
    assert STAGE_B_GENERATION_SETTINGS.output_reservation == 512


def test_stage_b_namespace_is_distinct_and_status_is_cheap(tmp_path: Path) -> None:
    result = generation_status(STAGE_B_GENERATOR_NAME, checkpoint_root=tmp_path / "checkpoints")

    assert result["generator"] == STAGE_B_GENERATOR_NAME
    assert result["model_loaded"] is False
    assert STAGE_B_CHECKPOINT_ROOT.name == STAGE_B_GENERATOR_NAME
    assert STAGE_B_RESULTS_ROOT.name == STAGE_B_GENERATOR_NAME
