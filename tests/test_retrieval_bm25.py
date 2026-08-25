import builtins

import pytest

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


def test_bm25_validates_parameters_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="Okapi"):
        BM25Index.build((chunk("a", "alpha"),), "arabic-raw-v1", k1=0)
    with pytest.raises(ValueError, match="Okapi"):
        BM25Index.build((chunk("a", "alpha"),), "arabic-raw-v1", b=2)
    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        BM25Index.build((chunk("a", "alpha"), chunk("a", "beta")), "arabic-raw-v1")


def test_bm25_reference_fallback_and_top_k_validation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real_import = builtins.__import__

    def without_bm25s(name: str, *args: object, **kwargs: object) -> object:
        if name == "bm25s":
            raise ImportError("test fallback")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_bm25s)
    index = BM25Index.build((chunk("a", "alpha"), chunk("b", "beta")), "arabic-raw-v1")

    assert index._backend is None
    assert index.search("alpha", top_k=1)[0].chunk_id == "a"
    with pytest.raises(ValueError, match="top_k"):
        index.search("alpha", top_k=0)
