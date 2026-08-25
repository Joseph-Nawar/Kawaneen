from __future__ import annotations

from kawaneen.extraction.provider import MockExtractionProvider
from kawaneen.retrieval.hybrid.contracts import SourceHit
from kawaneen.retrieval.models import RetrievalChunk


def _chunk(chunk_id: str, text: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_id="source-1",
        unit_type="article",
        display_text=text,
        search_text=text,
        source_unit_ids=(f"unit-{chunk_id}",),
        chunk_policy_hash="chunk-hash",
        normalization_policy_id="raw",
        normalization_policy_hash="normalization-hash",
        token_count=4,
    )


def test_serving_retriever_freezes_fusion_and_exposes_raw_logits() -> None:
    from kawaneen.retrieval.serving import HybridServingRetriever

    chunks = {f"c{i}": _chunk(f"c{i}", f"text {i}") for i in range(1, 4)}

    def sparse(query: str, top_k: int) -> tuple[SourceHit, ...]:
        assert query == "q"
        assert top_k == 50
        return tuple(SourceHit(f"c{i}", float(100 - i)) for i in (1, 2, 3))

    def dense(query: str, top_k: int) -> tuple[SourceHit, ...]:
        assert query == "q"
        assert top_k == 50
        return tuple(SourceHit(f"c{i}", float(200 - i)) for i in (3, 2, 1))

    retriever = HybridServingRetriever(
        chunks=chunks,
        sparse_search=sparse,
        dense_search=dense,
        reranker=lambda query, candidates: {
            item.chunk_id: float(10 - item.fused_rank) for item in candidates
        },
    )
    result = retriever.search("q", limit=2)

    assert [item.chunk_id for item in result.evidence] == ["c1", "c3"]
    assert result.evidence[0].score == 9.0
    assert result.evidence[0].score_type == "reranker_raw_logit"
    assert result.summary.fused_candidate_count == 20
    assert result.summary.reranker_depth == 8


def test_deterministic_extraction_never_calls_provider() -> None:
    from kawaneen.extraction.serving import ServingExtractor

    class ExplodingProvider:
        def propose(self, canonical_text: str, registry: object) -> object:
            raise AssertionError("deterministic mode must not call a provider")

    extractor = ServingExtractor(provider=ExplodingProvider())
    result = extractor.extract("يلتزم الطرف بالسداد.", mode="deterministic")

    assert result.capability_status == "operational_candidates"
    assert result.result.configuration == "deterministic-v1"
    assert result.result.source_provenance.source_id == "api-request"
    assert result.result.source_provenance.source_path == "request-body"


def test_hybrid_extraction_uses_fail_closed_assembly() -> None:
    from kawaneen.extraction.serving import ServingExtractor

    provider = MockExtractionProvider(
        {
            "schema_version": "phase11-proposal-v1",
            "rules": [
                {
                    "modality": "obligation",
                    "action": {"text": "يلتزم"},
                    "deadline_refs": ["T999"],
                }
            ],
        }
    )
    result = ServingExtractor(provider=provider).extract("يلتزم الطرف بالسداد.", mode="hybrid")

    assert provider.calls == 1
    assert result.capability_status == "experimental_limited"
    assert result.result.validation_metadata.proposal_valid is True
    assert any(
        diagnostic.code == "INVALID_CANDIDATE_REFERENCE"
        for diagnostic in result.result.validation_metadata.diagnostics
    )
    assert "PHASE11_HYBRID_EXPERIMENTAL_LIMITED" in result.warnings


def test_memory_corpus_orders_documents_and_hides_paths() -> None:
    from kawaneen.corpus.serving import InMemoryCorpusRepository, ServingDocument, ServingUnit

    repository = InMemoryCorpusRepository(
        (
            ServingDocument("b", "B", "source-b", (ServingUnit("b-u", "article", "B text"),)),
            ServingDocument("a", "A", "source-a", (ServingUnit("a-u", "article", "A text"),)),
        )
    )
    page = repository.list_documents(offset=0, limit=20)
    detail = repository.get_document("a")

    assert [item.document_id for item in page.items] == ["a", "b"]
    assert detail is not None
    assert not hasattr(detail, "path")
    assert repository.get_document("missing") is None
