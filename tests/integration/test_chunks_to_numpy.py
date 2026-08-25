from __future__ import annotations

import numpy as np
import pytest

from integration.test_pdf_to_chunks import FIXTURE, _corpus, _units_from_pdf
from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.chunking.strategies import build_chunks
from kawaneen.normalization import get_policy
from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.vector_index import NumpyExactIndex

pytestmark = pytest.mark.integration


def _retrieval_chunks() -> tuple[RetrievalChunk, ...]:
    units = _units_from_pdf(FIXTURE)
    chunks = []
    for chunk in build_chunks(
        units,
        _corpus(units),
        get_chunk_policy("legal-structure-v1"),
        get_policy("arabic-light-v1"),
    ):
        chunks.append(
            RetrievalChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.provenance["document_id"]
                if isinstance(chunk.provenance.get("document_id"), str)
                else units[0].document_id,
                source_id="phase14-synthetic",
                unit_type="article",
                display_text=chunk.display_text,
                search_text=chunk.search_text,
                source_unit_ids=chunk.source_unit_ids,
                chunk_policy_hash=chunk.chunk_policy_hash,
                normalization_policy_id=chunk.normalization_policy_id,
                normalization_policy_hash=chunk.normalization_policy_hash,
                token_count=chunk.token_count,
                source_spans=tuple((span.start, span.end) for span in chunk.source_spans),
            )
        )
    return tuple(chunks)


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
