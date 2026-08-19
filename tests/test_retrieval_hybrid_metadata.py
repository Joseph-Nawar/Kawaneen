from datetime import date

import numpy as np
import pytest

from kawaneen.retrieval.bm25 import BM25Index
from kawaneen.retrieval.hybrid.filtered import filtered_bm25, filtered_dense
from kawaneen.retrieval.hybrid.metadata import (
    DocumentMetadata,
    MetadataFilter,
    MetadataIndex,
    metadata_coverage,
)
from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.vector_index import NumpyExactIndex


def _chunk(chunk_id: str, text: str = "alpha") -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_id="source",
        unit_type="facts",
        display_text=text,
        search_text=text,
        source_unit_ids=(f"unit-{chunk_id}",),
        chunk_policy_hash="chunk",
        normalization_policy_id="arabic-light-v1",
        normalization_policy_hash="norm",
        token_count=1,
    )


def _records() -> tuple[DocumentMetadata, ...]:
    return (
        DocumentMetadata(
            document_id="doc-a",
            jurisdiction="SA",
            issuing_authority="MOJ",
            document_type="case",
            publication_date=date(2020, 1, 1),
            legal_status="active",
            regulation_name="Civil",
        ),
        DocumentMetadata(
            document_id="doc-b",
            jurisdiction="EG",
            issuing_authority="MOJ",
            document_type="statute",
            publication_date=date(2022, 6, 1),
            legal_status="repealed",
            regulation_name="Commercial",
        ),
        DocumentMetadata(document_id="doc-c"),
    )


def test_metadata_filter_supports_each_field_and_inclusive_dates() -> None:
    index = MetadataIndex.build(_records())

    assert index.eligible_ids(MetadataFilter(jurisdiction=("SA",))) == {"doc-a"}
    assert index.eligible_ids(MetadataFilter(issuing_authority=("MOJ",))) == {
        "doc-a",
        "doc-b",
    }
    assert index.eligible_ids(MetadataFilter(document_type=("statute",))) == {"doc-b"}
    assert index.eligible_ids(
        MetadataFilter(publication_date_from=date(2020, 1, 1), publication_date_to=date(2020, 1, 1))
    ) == {"doc-a"}
    assert index.eligible_ids(MetadataFilter(legal_status=("active",))) == {"doc-a"}
    assert index.eligible_ids(MetadataFilter(regulation_name=("Commercial",))) == {"doc-b"}


def test_metadata_filter_is_or_within_field_and_and_across_fields() -> None:
    index = MetadataIndex.build(_records())

    assert index.eligible_ids(
        MetadataFilter(jurisdiction=("SA", "EG"), document_type=("case",))
    ) == {"doc-a"}


def test_unknown_metadata_does_not_satisfy_constraints_and_empty_is_invalid() -> None:
    index = MetadataIndex.build(_records())
    assert index.eligible_ids(MetadataFilter(jurisdiction=("SA", "EG"))) == {"doc-a", "doc-b"}
    with pytest.raises(ValueError):
        MetadataFilter(jurisdiction=())
    with pytest.raises(ValueError):
        MetadataFilter(jurisdiction="SA")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MetadataFilter(publication_date_from="not-a-date")
    assert index.eligible_ids(MetadataFilter()) == {"doc-a", "doc-b", "doc-c"}


def test_filtered_bm25_and_dense_never_return_excluded_ids() -> None:
    chunks = (_chunk("a", "alpha"), _chunk("b", "alpha beta"), _chunk("c", "beta"))
    bm25 = BM25Index.build(chunks, "arabic-light-v1")
    assert [hit.chunk_id for hit in filtered_bm25(bm25, "alpha", {"doc-b"}, top_k=10)] == ["b"]

    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    dense = NumpyExactIndex.build(vectors, ("a", "b", "c"))
    assert [hit.chunk_id for hit in filtered_dense(dense, vectors[0], {"b"}, top_k=10)] == ["b"]


def test_coverage_report_is_text_free_and_counts_nulls() -> None:
    coverage = metadata_coverage(
        _records(), expected_document_ids=("doc-a", "doc-b", "doc-c", "doc-d")
    )
    assert coverage["fields"]["jurisdiction"]["populated_count"] == 2
    assert coverage["fields"]["jurisdiction"]["null_count"] == 2
    assert "SA" in coverage["fields"]["jurisdiction"]["distinct_values"]
    assert "document_id" not in str(coverage)
