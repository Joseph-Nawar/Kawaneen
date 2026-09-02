from __future__ import annotations

import numpy as np


def test_parity_gate_compares_ids_order_and_score_tolerance() -> None:
    from kawaneen.retrieval.qdrant_parity import compare_dense_indexes

    class Index:
        def search(self, query: np.ndarray, *, top_k: int):
            del query, top_k
            return (type("Hit", (), {"chunk_id": "a", "score": 1.0})(),)

    result = compare_dense_indexes(
        numpy_index=Index(),
        qdrant_index=Index(),
        queries=(
            ("query-b", np.asarray([1.0], dtype=np.float32)),
            ("query-a", np.asarray([1.0], dtype=np.float32)),
        ),
        sample_count=2,
    )
    assert result["provenance"] == "PHASE17_DEV"
    assert result["holdout_used"] is False
    assert result["sample_count"] == 2
    assert result["pass"] is True
    assert result["mismatched_query_count"] == 0
