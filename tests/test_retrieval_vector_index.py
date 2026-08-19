import numpy as np
import pytest

from kawaneen.retrieval.vector_index import (
    FaissExactIndex,
    NumpyExactIndex,
    validate_normalized_vectors,
)


def test_numpy_exact_index_returns_deterministic_inner_product_ranking() -> None:
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = NumpyExactIndex.build(vectors, ("b", "a", "c"))

    hits = index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=3)

    assert tuple(hit.chunk_id for hit in hits) == ("a", "b", "c")


def test_vector_validation_rejects_nan_and_non_normalized_rows() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_normalized_vectors(np.asarray([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        validate_normalized_vectors(np.asarray([[1.0, 0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        validate_normalized_vectors(np.asarray([[np.nan, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="normalized"):
        validate_normalized_vectors(np.asarray([[2.0, 0.0]], dtype=np.float32))


def test_vector_indexes_reject_duplicate_rows_and_query_dimension_errors() -> None:
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="unique chunk IDs"):
        NumpyExactIndex.build(vectors, ("same", "same"))
    index = NumpyExactIndex.build(vectors, ("chunk",))
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_faiss_and_numpy_rankings_match_when_faiss_is_available() -> None:
    pytest.importorskip("faiss")
    vectors = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = ("a", "b", "c")
    numpy_hits = NumpyExactIndex.build(vectors, ids).search(vectors[0], top_k=3)
    faiss_hits = FaissExactIndex.build(vectors, ids).search(vectors[0], top_k=3)
    assert tuple(hit.chunk_id for hit in numpy_hits) == tuple(hit.chunk_id for hit in faiss_hits)


def test_numpy_index_accepts_row_query_and_rejects_bad_topology() -> None:
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    index = NumpyExactIndex.build(vectors, ("chunk",))
    assert index.search(vectors[:1], top_k=1)[0].chunk_id == "chunk"
    with pytest.raises(ValueError, match="vector rows"):
        NumpyExactIndex.build(vectors, ())
