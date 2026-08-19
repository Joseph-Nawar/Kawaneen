from kawaneen.retrieval.bm25 import BM25Index
from kawaneen.retrieval.models import RetrievalChunk


def chunk(chunk_id: str, text: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        source_id="arabiccr",
        unit_type="reasoning",
        display_text=text,
        search_text=text,
        source_unit_ids=(chunk_id,),
        chunk_policy_hash="chunk",
        normalization_policy_id="arabic-raw-v1",
        normalization_policy_hash="norm",
        token_count=len(text.split()),
    )


def test_bm25_expected_ranking_on_tiny_corpus() -> None:
    index = BM25Index.build(
        (chunk("c", "alpha beta"), chunk("a", "alpha alpha"), chunk("b", "beta")),
        "arabic-raw-v1",
    )

    hits = index.search("alpha", top_k=3)

    assert tuple(hit.chunk_id for hit in hits) == ("a", "c", "b")
    assert hits[0].score > hits[1].score > hits[2].score
