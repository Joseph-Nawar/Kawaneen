from __future__ import annotations

from kawaneen.chunking.challenge import ChunkChallengeItem, PrivateChunkChallenge
from kawaneen.chunking.evaluation import map_gold_spans_to_chunks, run_chunking_ablation
from kawaneen.chunking.models import CitationAnchor, LegalChunk, SourceSpan


def _chunk(chunk_id: str, text: str, start: int, end: int) -> LegalChunk:
    span = SourceSpan("unit-1", start, end)
    return LegalChunk(
        chunk_id=chunk_id,
        strategy_id="legal-structure-v1",
        chunk_policy_hash="a" * 64,
        source_unit_ids=("unit-1",),
        display_text=text,
        search_text=text,
        source_spans=(span,),
        parent_id="parent-1",
        ancestor_ids=("doc-1", "parent-1"),
        sibling_ids=(),
        structure_path=("document", "section", "paragraph"),
        citation_anchor=CitationAnchor(kind="section", source_unit_id="unit-1"),
        token_count=len(text.split()),
        normalization_policy_id="arabic-light-v1",
        normalization_policy_hash="b" * 64,
        provenance={"source_id": "synthetic"},
    )


def test_gold_spans_map_to_overlapping_chunks_not_chunk_ids() -> None:
    chunks = (_chunk("chunk-a", "قانون مادة", 0, 10), _chunk("chunk-b", "حكم", 11, 15))
    assert map_gold_spans_to_chunks(chunks, (SourceSpan("unit-1", 3, 6),)) == ("chunk-a",)


def test_chunking_ablation_reports_retrieval_citation_and_context_metrics() -> None:
    chunks = (_chunk("chunk-a", "قانون مادة", 0, 10), _chunk("chunk-b", "حكم", 11, 15))
    item = ChunkChallengeItem(
        query_id="q1",
        slice_name="local_passage",
        query_text="قانون",
        document_id="doc-1",
        gold_spans=(SourceSpan("unit-1", 3, 6),),
        construction_version="phase5-chunk-challenge-v1",
    )
    challenge = PrivateChunkChallenge(
        seed=1,
        construction_version="phase5-chunk-challenge-v1",
        items=(item,),
        qrels={"q1": item.gold_spans},
    )
    report = run_chunking_ablation({"legal-structure-v1": chunks}, challenge)
    assert report.strategy_metrics["legal-structure-v1"]["recall_at_1"] == 1.0
    assert report.citation_metrics["legal-structure-v1"]["citation_precision_at_1"] == 0.3
    assert report.context_metrics["legal-structure-v1"]["context_coverage_at_1"] == 1.0
