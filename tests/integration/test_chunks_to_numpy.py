from __future__ import annotations

import numpy as np
import pytest
from phase14_support import build_phase14_stack

from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.vector_index import NumpyExactIndex

pytestmark = pytest.mark.integration


def _retrieval_chunks() -> tuple[RetrievalChunk, ...]:
    return build_phase14_stack().chunks


def test_chunks_have_exact_numpy_index_correspondence_and_deterministic_ranking() -> None:
    chunks = _retrieval_chunks()
    vectors = np.eye(len(chunks), dtype=np.float32)
    index = NumpyExactIndex.build(vectors, [chunk.chunk_id for chunk in chunks])

    result = index.search(vectors[0], top_k=len(chunks))
    repeat = index.search(vectors[0], top_k=len(chunks))

    assert result[0].chunk_id == chunks[0].chunk_id
    assert result == repeat
    assert {hit.chunk_id for hit in result} == {chunk.chunk_id for chunk in chunks}
    assert all(vector.dtype == np.float32 for vector in index.vectors)


@pytest.mark.parametrize(
    ("vectors", "ids", "message"),
    [
        (np.ones((2, 3), dtype=np.float32), ["a", "b"], "normalized"),
        (np.eye(2, dtype=np.float32), ["a", "a"], "unique"),
    ],
)
def test_numpy_index_rejects_invalid_dimensions_or_duplicate_ids(
    vectors: np.ndarray, ids: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NumpyExactIndex.build(vectors, ids)


def test_numpy_index_rejects_query_dimension_mismatch() -> None:
    index = NumpyExactIndex.build(np.eye(2, dtype=np.float32), ["a", "b"])

    with pytest.raises(ValueError, match="dimension"):
        index.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
