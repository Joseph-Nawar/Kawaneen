from kawaneen.retrieval.keyword import KeywordIndex
from kawaneen.retrieval.models import RetrievalChunk


def chunk(chunk_id: str, text: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        source_id="alarb",
        unit_type="facts",
        display_text=text,
        search_text=text,
        source_unit_ids=(chunk_id,),
        chunk_policy_hash="chunk",
        normalization_policy_id="arabic-raw-v1",
        normalization_policy_hash="norm",
        token_count=2,
    )


def test_keyword_jaccard_ranks_overlap_and_breaks_ties_by_chunk_id() -> None:
    index = KeywordIndex.build(
        (chunk("b", "alpha beta"), chunk("a", "alpha gamma"), chunk("c", "delta")),
        "arabic-raw-v1",
    )

    hits = index.search("alpha", top_k=3)

    assert tuple(hit.chunk_id for hit in hits) == ("a", "b", "c")
    assert hits[0].score == hits[1].score == 0.5


def test_keyword_rejects_duplicate_chunk_ids() -> None:
    try:
        KeywordIndex.build((chunk("a", "one"), chunk("a", "two")), "arabic-raw-v1")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate chunk IDs must be rejected")
