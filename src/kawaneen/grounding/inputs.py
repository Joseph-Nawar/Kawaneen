"""Read-only readers for the frozen Phase-8 DEV output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from kawaneen.grounding.contracts import RetrievalInput

PHASE8_SELECTION_SHA256 = "a62cc772f2b71883355c7935da7e7b87ab4d22b3746553148b4f64ef20f28b0b"
PHASE8_SELECTION = Path("data/manifests/retrieval/phase8_dev_selection.json")
PHASE8_RERANK_ROOT = Path("artifacts/private/phase8_retrieval/rerank")


def load_frozen_phase8_dev_rankings(
    *,
    selection_path: Path = PHASE8_SELECTION,
    rerank_root: Path = PHASE8_RERANK_ROOT,
    expected_selection_sha256: str = PHASE8_SELECTION_SHA256,
) -> tuple[RetrievalInput, ...]:
    """Load exactly the persisted serving top-8; never run retrieval."""

    actual_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    if actual_sha != expected_selection_sha256:
        raise ValueError("Phase-8 DEV selection SHA does not match the frozen input")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError("Phase-8 DEV selection is not an object")
    selection = cast(dict[str, object], selection)
    if selection.get("status") != "phase8_dev_selection_frozen":
        raise ValueError("Phase-8 DEV selection is not frozen")
    if selection.get("selected_pipeline") != "rrf_reranked":
        raise ValueError("Phase-8 selected pipeline is not the frozen reranked pipeline")
    reranker = selection.get("reranker")
    if not isinstance(reranker, dict):
        raise ValueError("Phase-8 serving depth is not 8")
    typed_reranker = cast(dict[str, object], reranker)
    if typed_reranker.get("serving_depth") != 8:
        raise ValueError("Phase-8 serving depth is not 8")

    manifest = json.loads((rerank_root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Phase-8 rerank manifest is not an object")
    manifest = cast(dict[str, object], manifest)
    queries_value = manifest.get("queries")
    if not isinstance(queries_value, dict):
        raise ValueError("Phase-8 rerank manifest has no query table")
    queries = cast(dict[str, object], queries_value)
    result: list[RetrievalInput] = []
    for query_id in sorted(str(value) for value in queries):
        entry_value = queries[query_id]
        if not isinstance(entry_value, dict):
            raise ValueError(f"Phase-8 query entry is not an object: {query_id}")
        entry = cast(dict[str, object], entry_value)
        if entry.get("status") != "completed":
            raise ValueError(f"Phase-8 query is not completed: {query_id}")
        path_name = entry.get("path")
        if not isinstance(path_name, str) or Path(path_name).name != path_name:
            raise ValueError(f"unsafe Phase-8 ranking path: {query_id}")
        payload = json.loads((rerank_root / path_name).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Phase-8 ranking is not an object: {query_id}")
        payload = cast(dict[str, object], payload)
        if payload.get("query_id") != query_id:
            raise ValueError(f"Phase-8 ranking query ID mismatch: {query_id}")
        ranked_value = payload.get("ranked_chunk_ids")
        if not isinstance(ranked_value, list):
            raise ValueError(f"Phase-8 ranking has fewer than 8 results: {query_id}")
        ranked = cast(list[object], ranked_value)
        if len(ranked) < 8:
            raise ValueError(f"Phase-8 ranking has fewer than 8 results: {query_id}")
        top_eight = tuple(str(value) for value in ranked[:8])
        if len(set(top_eight)) != len(top_eight):
            raise ValueError(f"duplicate Phase-8 top-8 chunk IDs: {query_id}")
        result.extend(
            RetrievalInput(query_id=query_id, rank=rank, chunk_id=chunk_id)
            for rank, chunk_id in enumerate(top_eight, start=1)
        )
    return tuple(result)
