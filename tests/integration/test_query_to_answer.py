from __future__ import annotations

import json
from pathlib import Path

import pytest

from integration.test_chunks_to_numpy import _retrieval_chunks
from integration.test_pdf_to_chunks import FIXTURE, _corpus, _units_from_pdf
from kawaneen.generation.policy import PolicyOutcome
from kawaneen.generation.serving import ServingAnswerer
from kawaneen.grounding.assembly import ContextAssembler
from kawaneen.grounding.contracts import (
    CitationRequest,
    ClaimDraft,
    ContextPack,
    GeneratedDraft,
    RetrievalInput,
)
from kawaneen.grounding.provenance import CanonicalCorpusResolver
from kawaneen.grounding.verification import verify_draft
from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
from kawaneen.retrieval.serving import HybridServingRetriever, ServingRetrievalResult

pytestmark = pytest.mark.integration


def _retriever() -> tuple[HybridServingRetriever, list[tuple[str, int]]]:
    chunks = _retrieval_chunks()
    calls: list[tuple[str, int]] = []

    def sparse(query: str, top_k: int) -> tuple[SourceHit, ...]:
        calls.append((f"sparse:{query}", top_k))
        return tuple(
            SourceHit(chunk.chunk_id, float(len(chunks) - index))
            for index, chunk in enumerate(chunks)
        )

    def dense(query: str, top_k: int) -> tuple[SourceHit, ...]:
        calls.append((f"dense:{query}", top_k))
        return tuple(SourceHit(chunk.chunk_id, float(index)) for index, chunk in enumerate(chunks))

    def rerank(query: str, candidates: tuple[object, ...]) -> dict[str, float]:
        return {candidate.chunk_id: 100.0 - index for index, candidate in enumerate(candidates)}  # type: ignore[attr-defined]

    return (
        HybridServingRetriever(
            chunks={chunk.chunk_id: chunk for chunk in chunks},
            sparse_search=sparse,
            dense_search=dense,
            reranker=rerank,
            fusion_config=FusionConfig(sparse_weight=1.0, dense_weight=0.25),
        ),
        calls,
    )


def test_query_retrieval_uses_frozen_serving_depths_and_raw_reranker_scores() -> None:
    retriever, calls = _retriever()

    result = retriever.search("ما مهلة الاعتراض؟", limit=8)

    assert calls == [("sparse:ما مهلة الاعتراض؟", 50), ("dense:ما مهلة الاعتراض؟", 50)]
    assert result.summary.hit_count == 3
    assert result.summary.returned_count == 3
    assert result.summary.score_type == "reranker_raw_logit"
    assert result.evidence[0].score == 100.0
    assert result.evidence[0].provenance in {"sparse-only", "dense-only", "both"}


def _write_resolver_inputs(tmp_path: Path):
    units = _units_from_pdf(FIXTURE)
    from kawaneen.chunking.policies import get_chunk_policy
    from kawaneen.chunking.strategies import build_chunks
    from kawaneen.normalization import get_policy

    chunks = build_chunks(
        units,
        _corpus(units),
        get_chunk_policy("legal-structure-v1"),
        get_policy("arabic-light-v1"),
    )
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "summary": {"corpus_hash": "b" * 64},
                "units": [unit.model_dump(mode="json") for unit in units],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(
            json.dumps(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_unit_ids": list(chunk.source_unit_ids),
                    "source_spans": [
                        {"unit_id": span.unit_id, "start": span.start, "end": span.end}
                        for span in chunk.source_spans
                    ],
                    "chunk_policy_hash": chunk.chunk_policy_hash,
                },
                ensure_ascii=False,
            )
            + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )
    documents = tmp_path / "documents.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "document_id": units[0].document_id,
                    "title": "Synthetic Appeals Regulation",
                    "source_id": "phase14-synthetic",
                    "raw_article_label": "المادة ١٢",
                    "source_metadata_json": "{}",
                }
            ]
        ),
        documents,
    )
    return (
        units,
        chunks,
        CanonicalCorpusResolver.from_json(canonical, chunks_path, document_paths=(documents,)),
    )


def test_query_to_grounded_answer_and_abstention_stop_before_generation(tmp_path: Path) -> None:
    units, chunks, resolver = _write_resolver_inputs(tmp_path)
    retriever, _ = _retriever()

    class Counter:
        identity = "phase14"

        def count(self, text: str) -> int:
            return len(text)

    assembler = ContextAssembler(resolver, Counter(), max_context_tokens=10_000)
    calls = 0

    def build_context(query: str, retrieval: ServingRetrievalResult) -> ContextPack:
        ranked = tuple(
            RetrievalInput(query_id="q1", rank=index, chunk_id=evidence.chunk_id)
            for index, evidence in enumerate(retrieval.evidence, start=1)
        )
        return assembler.assemble(
            query_id="q1",
            ranked_inputs=ranked,
            phase8_selection_sha256="a" * 64,
            canonical_corpus_hash="b" * 64,
        )

    def generate(query: str, context: ContextPack) -> GeneratedDraft:
        nonlocal calls
        calls += 1
        evidence = context.evidence[0]
        return GeneratedDraft(
            answer_text=evidence.display_text,
            claims=(
                ClaimDraft(
                    claim_id="C001",
                    claim_text=evidence.display_text,
                    citations=(
                        CitationRequest(
                            evidence_id=evidence.evidence_id,
                            quoted_text=evidence.display_text,
                        ),
                    ),
                ),
            ),
        )

    answerer = ServingAnswerer(
        retriever=lambda query, limit=8: retriever.search(query, limit),
        context_builder=build_context,
        policy_evaluator=lambda query, context: PolicyOutcome(allowed="مهلة" in query),
        generator=generate,
        verifier=lambda context, draft: verify_draft(context, draft, resolver),
    )

    grounded = answerer.answer("ما مهلة الاعتراض؟")
    assert grounded.answerable is True
    assert grounded.citations
    assert grounded.citations[0].document_id == units[0].document_id
    assert grounded.citations[0].article == "المادة ١٢"
    assert grounded.citations[0].quoted_text == units[0].text
    assert calls == 1

    abstained = answerer.answer("ما لون السماء؟")
    assert abstained.answerable is False
    assert abstained.citations == ()
    assert calls == 1
    assert len(chunks) == 3
