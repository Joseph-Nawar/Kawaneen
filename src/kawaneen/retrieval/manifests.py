# pyright: basic
"""Deterministic text-free Phase 7 manifest helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from os import PathLike

from kawaneen.retrieval.models import RetrievalChunk


def stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hash_file(path: str | PathLike[str]) -> str:
    from pathlib import Path

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_corpus_manifest(
    chunks: Iterable[RetrievalChunk], *, corpus_hash: str, release_hash: str = ""
) -> dict[str, object]:
    selected = tuple(chunks)
    chunk_ids = sorted(chunk.chunk_id for chunk in selected)
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("duplicate chunk IDs are not allowed")
    return {
        "schema_version": 1,
        "status": "phase7_retrieval_corpus_ready",
        "corpus_hash": corpus_hash,
        "release_hash": release_hash,
        "chunk_policy_id": "legal-structure-v1",
        "chunk_count": len(selected),
        "chunk_ids_hash": stable_hash(chunk_ids),
        "document_count": len({chunk.document_id for chunk in selected}),
        "source_counts": {
            source: sum(chunk.source_id == source for chunk in selected)
            for source in sorted({chunk.source_id for chunk in selected})
        },
        "chunk_policy_hashes": sorted({chunk.chunk_policy_hash for chunk in selected}),
        "normalization_policy_hashes": sorted(
            {chunk.normalization_policy_hash for chunk in selected}
        ),
    }
