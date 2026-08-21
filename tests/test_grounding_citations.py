from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kawaneen.grounding.assembly import ContextAssembler
from kawaneen.grounding.contracts import CitationRequest, ClaimDraft, GeneratedDraft, RetrievalInput
from kawaneen.grounding.provenance import CanonicalCorpusResolver
from kawaneen.grounding.verification import verify_citation, verify_draft


class FakeCounter:
    identity = "fake-v1"

    def count(self, text: str) -> int:
        return len(text)


def pack(tmp_path: Path):
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "summary": {"corpus_hash": "b" * 64},
                "units": [
                    {
                        "unit_id": "u-1",
                        "document_id": "doc-1",
                        "ordinal": 1,
                        "text": "النص الأصلي ِ",
                        "unit_type": "events",
                        "provenance": {
                            "source_id": "canonical-source",
                            "source_version": "v1",
                            "source_path": "private",
                            "source_row": 1,
                            "source_field": "events",
                            "split": "",
                        },
                    },
                    {
                        "unit_id": "u-2",
                        "document_id": "doc-2",
                        "ordinal": 1,
                        "text": "النص الأصلي ِ",
                        "unit_type": "events",
                        "provenance": {
                            "source_id": "other-source",
                            "source_version": "v1",
                            "source_path": "private",
                            "source_row": 2,
                            "source_field": "events",
                            "split": "",
                        },
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "".join(
            json.dumps(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "display_text": "forged",
                    "source_unit_ids": [unit_id],
                    "source_spans": [],
                }
            )
            + "\n"
            for chunk_id, document_id, unit_id in (
                ("chunk-1", "doc-1", "u-1"),
                ("chunk-2", "doc-2", "u-2"),
            )
        ),
        encoding="utf-8",
    )
    resolver = CanonicalCorpusResolver.from_json(canonical, chunks)
    return resolver, ContextAssembler(resolver, FakeCounter(), max_context_tokens=1000).assemble(
        query_id="q1",
        ranked_inputs=(RetrievalInput(query_id="q1", rank=1, chunk_id="chunk-1"),),
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
    )


def test_accepted_citation_is_authoritative_and_exact(tmp_path: Path) -> None:
    resolver, context = pack(tmp_path)
    request = CitationRequest(evidence_id="E001", quoted_text="النص الأصلي ِ")
    result = verify_citation(context, request, resolver)
    assert result.valid is True
    assert result.citation is not None
    assert result.citation.document_id == "doc-1"
    assert result.citation.chunk_id == "chunk-1"
    assert result.citation.quoted_text in resolver.units_by_id["u-1"].display_text
    assert result.citation.document_title is None
    assert result.citation.jurisdiction is None
    assert result.citation.article is None
    assert result.citation.page is None
    assert result.citation.source_url is None


@pytest.mark.parametrize(
    ("evidence_id", "quoted_text"),
    [
        ("E999", "النص الأصلي ِ"),
        ("E001", "نص من خارج السياق"),
        ("E001", "النص المعدل ِ"),
        ("E001", "النص الاصلي ِ"),
        ("E001", ""),
    ],
)
def test_adversarial_citations_are_rejected(
    tmp_path: Path, evidence_id: str, quoted_text: str
) -> None:
    resolver, context = pack(tmp_path)
    result = verify_citation(
        context,
        CitationRequest(evidence_id=evidence_id, quoted_text=quoted_text),
        resolver,
    )
    assert result.valid is False
    assert result.citation is None


def test_out_of_context_source_with_identical_text_is_rejected(tmp_path: Path) -> None:
    resolver, context = pack(tmp_path)
    request = CitationRequest(evidence_id="E002", quoted_text="النص الأصلي ِ")
    result = verify_citation(context, request, resolver)
    assert result.valid is False


def test_generator_cannot_invent_page_article_chunk_or_source_url(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        CitationRequest(
            evidence_id="E001",
            quoted_text="النص الأصلي ِ",
            page="99",  # type: ignore[call-arg]
            article="invented",  # type: ignore[call-arg]
            chunk_id="forged",  # type: ignore[call-arg]
            source_url="https://forged.example",  # type: ignore[call-arg]
        )


def test_unsupported_claim_and_empty_context_force_abstention(tmp_path: Path) -> None:
    resolver, context = pack(tmp_path)
    unsupported = verify_draft(
        context,
        GeneratedDraft(
            answer_text="إجابة أخرى",
            claims=(
                ClaimDraft(
                    claim_id="C001",
                    claim_text="النص غير ممثل",
                    citations=(CitationRequest(evidence_id="E001", quoted_text="النص الأصلي ِ"),),
                ),
            ),
        ),
        resolver,
    )
    assert unsupported.should_abstain is True
    assert unsupported.unsupported_claims == ("C001",)

    empty = ContextAssembler(resolver, FakeCounter(), max_context_tokens=1000).assemble(
        query_id="q-empty",
        ranked_inputs=(),
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
    )
    result = verify_draft(empty, GeneratedDraft(answer_text="إجابة", claims=()), resolver)
    assert result.should_abstain is True


def test_malformed_draft_and_missing_claim_citation_are_rejected(tmp_path: Path) -> None:
    resolver, context = pack(tmp_path)
    with pytest.raises(ValidationError):
        GeneratedDraft(answer_text="إجابة", claims=("malformed",))  # type: ignore[arg-type]
    result = verify_draft(
        context,
        GeneratedDraft(
            answer_text="النص غير ممثل",
            claims=(ClaimDraft(claim_id="C001", claim_text="النص غير ممثل", citations=()),),
        ),
        resolver,
    )
    assert result.should_abstain is True
