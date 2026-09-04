from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def select_dev_query_records(
    records: Sequence[Mapping[str, object]], *, sample_count: int = 20
) -> tuple[Mapping[str, object], ...]:
    """Select stable DEV records by ID hash, without consulting parity results."""

    dev = tuple(
        record
        for record in records
        if str(record.get("split", "")).lower() == "dev" and isinstance(record.get("query_id"), str)
    )
    if len(dev) < sample_count:
        raise ValueError("fewer than the requested number of DEV queries are available")
    return tuple(
        sorted(
            dev,
            key=lambda record: hashlib.sha256(str(record["query_id"]).encode("utf-8")).hexdigest(),
        )[:sample_count]
    )


def compare_dense_indexes(
    *,
    numpy_index: Any,
    qdrant_index: Any,
    queries: Sequence[tuple[str, np.ndarray]],
    sample_count: int = 20,
    top_k: int = 50,
    tolerance: float = 1e-5,
) -> dict[str, object]:
    """Compare exact indexes over a stable DEV-only query selection."""

    selected = tuple(sorted(queries, key=lambda item: item[0])[:sample_count])
    mismatched = 0
    max_error = 0.0
    for _query_id, vector in selected:
        numpy_hits = numpy_index.search(vector, top_k=top_k)
        qdrant_hits = qdrant_index.search(vector, top_k=top_k)
        numpy_ids = tuple(hit.chunk_id for hit in numpy_hits)
        qdrant_ids = tuple(hit.chunk_id for hit in qdrant_hits)
        score_error = max(
            (
                abs(float(left.score) - float(right.score))
                for left, right in zip(numpy_hits, qdrant_hits, strict=False)
            ),
            default=0.0,
        )
        max_error = max(max_error, score_error)
        if (
            numpy_ids != qdrant_ids
            or len(numpy_hits) != len(qdrant_hits)
            or score_error > tolerance
        ):
            mismatched += 1
    return {
        "schema": "phase17-qdrant-parity-v1",
        "provenance": "PHASE17_DEV",
        "holdout_used": False,
        "sample_count": len(selected),
        "top_k": top_k,
        "pass": mismatched == 0,
        "mismatched_query_count": mismatched,
        "max_score_error": max_error,
        "numpy_backend": "NumpyExactIndex",
        "qdrant_mode": "exact",
    }


__all__ = ["compare_dense_indexes", "select_dev_query_records"]
