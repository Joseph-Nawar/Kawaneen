from __future__ import annotations

import json
from pathlib import Path

from kawaneen.generation.contracts import (
    AbstentionReason,
    ClaimMode,
    GenerationDecision,
    GenerationResult,
    ModelOutputCitation,
    ModelOutputClaim,
)
from kawaneen.generation.postprocessing import finalize_generation
from kawaneen.grounding.contracts import ContextPack, EvidenceReference, SourceRecord
from kawaneen.grounding.provenance import CanonicalCorpusResolver


def resolver(tmp_path: Path) -> CanonicalCorpusResolver:
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "document_id": "doc-1",
                        "ordinal": 1,
                        "text": "The deadline is thirty days.",
                        "unit_type": "events",
                        "provenance": {
                            "source_id": "source-1",
                            "source_version": "v1",
                            "source_path": "private",
                            "source_row": 1,
                            "source_field": "events",
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
        json.dumps({"chunk_id": "c1", "source_unit_ids": ["u1"], "source_spans": []}) + "\n",
        encoding="utf-8",
    )
    return CanonicalCorpusResolver.from_json(canonical, chunks)


def pack() -> ContextPack:
    source = SourceRecord(document_id="doc-1", document_title="Law")
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
        evidence=(
            EvidenceReference(
                evidence_id="E001",
                unit_id="u1",
                block_id="B001",
                document_id="doc-1",
                display_text="The deadline is thirty days.",
                source=source,
                contributing_chunk_ids=("c1",),
                contributing_ranks=(1,),
            ),
        ),
        omissions=(),
    )


def claim(
    text: str,
    quote: str,
    *,
    mode: ClaimMode = ClaimMode.DIRECT,
) -> ModelOutputClaim:
    return ModelOutputClaim(
        mode=mode,
        text=text,
        citations=(ModelOutputCitation(evidence_id="E001", quoted_text=quote),),
    )


def test_exact_extractive_result_can_be_rendered_after_phase9_verification(tmp_path: Path) -> None:
    result = finalize_generation(
        pack(),
        GenerationResult(
            decision=GenerationDecision.ANSWER,
            claims=(claim("The deadline is thirty days.", "The deadline is thirty days."),),
        ),
        resolver(tmp_path),
        jurisdiction_text="Saudi Arabia",
        disclaimer_text="General information only.",
    )

    assert result.rendered_answer is not None
    assert "The deadline is thirty days." in result.rendered_answer
    assert result.abstention_reason is None


def test_non_exact_claim_is_not_finalized_when_semantic_support_is_deferred(tmp_path: Path) -> None:
    result = finalize_generation(
        pack(),
        GenerationResult(
            decision=GenerationDecision.ANSWER,
            claims=(
                claim(
                    "The deadline may be extended.",
                    "The deadline is thirty days.",
                    mode=ClaimMode.INTERPRETATION,
                ),
            ),
        ),
        resolver(tmp_path),
    )

    assert result.rendered_answer is None
    assert result.abstention_reason is AbstentionReason.SEMANTIC_SUPPORT_UNAVAILABLE
