from __future__ import annotations

import numpy as np
import pytest


class _Point:
    def __init__(self, point_id: int, score: float, chunk_id: str) -> None:
        self.id = point_id
        self.score = score
        self.payload = {"chunk_id": chunk_id}


class _Client:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def query_points(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return type("Response", (), {"points": (_Point(2, 0.8, "b"), _Point(1, 0.8, "a"))})()


def test_qdrant_exact_index_uses_exact_search_and_deterministic_ties() -> None:
    from kawaneen.retrieval.qdrant_index import QdrantExactIndex

    client = _Client()
    index = QdrantExactIndex.build(
        client=client,
        collection_name="kawaneen_fixture_bge_m3",
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        chunk_ids=("a", "b"),
    )

    result = index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)

    assert [item.chunk_id for item in result] == ["a", "b"]
    assert client.calls[0]["collection_name"] == "kawaneen_fixture_bge_m3"
    assert client.calls[0]["limit"] == 2
    assert client.calls[0]["search_params"].exact is True


def test_qdrant_exact_index_rejects_invalid_query_and_malformed_hits() -> None:
    from kawaneen.retrieval.qdrant_index import QdrantExactIndex

    class BadClient:
        def query_points(self, **kwargs: object) -> object:
            del kwargs
            return type("Response", (), {"points": (object(),)})()

    index = QdrantExactIndex.build(
        client=BadClient(),
        collection_name="fixture",
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        chunk_ids=("a",),
    )
    with pytest.raises(ValueError, match="normalized"):
        index.search(np.asarray([2.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="payload"):
        index.search(np.asarray([1.0, 0.0], dtype=np.float32))
