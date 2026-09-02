# pyright: basic
"""Idempotent seeding of the Phase 17-owned exact Qdrant collection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from kawaneen.retrieval.serving import load_serving_chunks
from kawaneen.retrieval.vector_index import validate_normalized_vectors

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def collection_name_for(corpus_hash: str) -> str:
    if _SHA256.fullmatch(corpus_hash) is None:
        raise ValueError("corpus hash must be a lowercase SHA-256 value")
    return f"kawaneen_{corpus_hash[:12]}_bge_m3"


@dataclass(frozen=True, slots=True)
class QdrantSeed:
    collection_name: str
    corpus_hash: str
    model_revision: str
    vectors: np.ndarray
    chunk_ids: tuple[str, ...]
    chunks: dict[str, object]


def load_qdrant_seed(
    root: Path,
    *,
    expected_corpus_hash: str,
    expected_model_revision: str,
    expected_dimension: int = 1024,
) -> QdrantSeed:
    """Load and validate only the frozen Phase 7 serving inputs."""

    chunks = load_serving_chunks(root / "corpus" / "chunks.jsonl")
    embedding_root = root / "embeddings" / "BAAI__bge-m3" / "arabic-raw-v1"
    vectors_paths = sorted(embedding_root.glob("*/vectors.npy"))
    if len(vectors_paths) != 1:
        raise ValueError("frozen BGE-M3 vector asset is missing or ambiguous")
    vectors_path = vectors_paths[0]
    ids_path = vectors_path.with_name("ids.json")
    try:
        vectors = np.load(vectors_path, allow_pickle=False)
        raw_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("frozen Qdrant seed assets are invalid") from error
    values = np.asarray(vectors)
    validate_normalized_vectors(values)
    if values.shape[1] != expected_dimension:
        raise ValueError("frozen vector dimension does not match the serving manifest")
    if not _SHA256.fullmatch(expected_corpus_hash):
        raise ValueError("frozen corpus identity is invalid")
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("frozen serving manifest is invalid") from error
        if not isinstance(manifest, dict) or manifest.get("corpus_hash") != expected_corpus_hash:
            raise ValueError("frozen corpus identity does not match the serving manifest")
    if not expected_model_revision:
        raise ValueError("frozen model revision is missing")
    if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
        raise ValueError("frozen Qdrant IDs are invalid")
    chunk_ids = tuple(cast(str, item) for item in raw_ids)
    if len(chunk_ids) != len(set(chunk_ids)) or len(chunk_ids) != len(chunks):
        raise ValueError("frozen Qdrant IDs do not match the serving corpus")
    if set(chunk_ids) != set(chunks):
        raise ValueError("frozen Qdrant IDs do not match the serving corpus")
    return QdrantSeed(
        collection_name=collection_name_for(expected_corpus_hash),
        corpus_hash=expected_corpus_hash,
        model_revision=expected_model_revision,
        vectors=values.astype(np.float32, copy=True),
        chunk_ids=chunk_ids,
        chunks=cast(dict[str, object], chunks),
    )


def _models() -> Any:
    try:
        from qdrant_client.http import models
    except ImportError as error:
        raise RuntimeError("qdrant-client is required to seed Qdrant") from error
    return models


def _collection_matches(client: Any, seed: QdrantSeed) -> bool:
    try:
        info = client.get_collection(seed.collection_name)
        config = info.config.params.vectors
        count = client.count(collection_name=seed.collection_name, exact=True).count
        points, _ = client.scroll(collection_name=seed.collection_name, limit=1, with_payload=True)
    except (AttributeError, RuntimeError, ValueError):
        return False
    size = getattr(config, "size", None)
    distance = str(getattr(config, "distance", "")).lower()
    if not (
        size == seed.vectors.shape[1] and "cosine" in distance and count == len(seed.chunk_ids)
    ):
        return False
    if not points:
        return False
    payload = getattr(points[0], "payload", None)
    return (
        isinstance(payload, dict)
        and payload.get("corpus_hash") == seed.corpus_hash
        and payload.get("model_revision") == seed.model_revision
    )


def seed_qdrant_collection(client: Any, seed: QdrantSeed) -> str:
    """Create or repair only the deterministic Phase-17-owned collection."""

    if _collection_matches(client, seed):
        return seed.collection_name
    try:
        client.get_collection(seed.collection_name)
    except Exception:
        pass
    else:
        client.delete_collection(seed.collection_name)
    models = _models()
    client.create_collection(
        collection_name=seed.collection_name,
        vectors_config=models.VectorParams(
            size=int(seed.vectors.shape[1]),
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(m=0),
        ),
    )
    points = [
        models.PointStruct(
            id=index,
            vector=vector.tolist(),
            payload={
                "chunk_id": chunk_id,
                "corpus_hash": seed.corpus_hash,
                "model_revision": seed.model_revision,
            },
        )
        for index, (chunk_id, vector) in enumerate(zip(seed.chunk_ids, seed.vectors, strict=True))
    ]
    client.upsert(collection_name=seed.collection_name, points=points, wait=True)
    return seed.collection_name


__all__ = ["QdrantSeed", "collection_name_for", "load_qdrant_seed", "seed_qdrant_collection"]
