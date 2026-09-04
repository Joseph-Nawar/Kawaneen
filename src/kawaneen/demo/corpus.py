from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.vector_index import validate_normalized_vectors

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DEFAULT_DEMO_ROOT = Path("data/demo")


@dataclass(frozen=True, slots=True)
class DemoCorpus:
    chunks: dict[str, RetrievalChunk]
    metadata: dict[str, dict[str, str | None]]
    vectors: np.ndarray
    manifest: dict[str, object]


def load_demo_corpus(root: Path = DEFAULT_DEMO_ROOT) -> DemoCorpus:
    manifest = _object(root / "manifest.json")
    if manifest.get("synthetic") is not True or manifest.get("jurisdiction") != "KAWANEEN_DEMO":
        raise ValueError("demo corpus must be explicitly synthetic KAWANEEN_DEMO content")
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("demo corpus uses an unexpected locked embedding model")
    chunks, metadata, chunks_bytes = _load_chunks(root / "corpus" / "chunks.jsonl")
    try:
        ids = json.loads((root / "ids.json").read_text(encoding="utf-8"))
        vectors = np.load(root / "vectors.npy", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("demo embedding assets are invalid") from error
    values = np.asarray(vectors)
    validate_normalized_vectors(values)
    if values.shape[1] != manifest.get("embedding_dimension") or values.shape[1] != 384:
        raise ValueError("demo embedding dimension is invalid")
    if values.shape[0] != manifest.get("vector_count") or values.shape[0] != len(chunks):
        raise ValueError("demo vector count does not match the corpus")
    ids_list = cast(list[object], ids) if isinstance(ids, list) else None
    if ids_list is None or any(not isinstance(item, str) for item in ids_list):
        raise ValueError("demo IDs are invalid")
    if tuple(cast(list[str], ids_list)) != tuple(chunks):
        raise ValueError("demo IDs do not match the corpus ordering")
    if hashlib.sha256(chunks_bytes).hexdigest() != manifest.get("corpus_sha256"):
        raise ValueError("demo corpus hash does not match its manifest")
    vector_bytes = (root / "vectors.npy").read_bytes()
    if hashlib.sha256(vector_bytes).hexdigest() != manifest.get("embedding_sha256"):
        raise ValueError("demo embedding hash does not match its manifest")
    return DemoCorpus(chunks, metadata, values.astype(np.float32, copy=True), manifest)


def _object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("demo manifest is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise ValueError("demo manifest must be an object")
    return cast(dict[str, object], value)


def _load_chunks(
    path: Path,
) -> tuple[dict[str, RetrievalChunk], dict[str, dict[str, str | None]], bytes]:
    try:
        raw_bytes = path.read_bytes()
        lines = raw_bytes.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("demo chunks are unavailable or invalid") from error
    chunks: dict[str, RetrievalChunk] = {}
    metadata: dict[str, dict[str, str | None]] = {}
    for line in lines:
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("demo chunk must be an object")
        row = cast(dict[str, object], value)
        chunk_id = _required(row, "chunk_id")
        if chunk_id in chunks:
            raise ValueError("demo chunk IDs must be unique")
        if any(token in chunk_id.casefold() for token in ("saudi", "moj", "private")):
            raise ValueError("demo chunk contains a private-source identifier")
        text = _required(row, "display_text")
        title = _required(row, "document_title")
        if not all(
            marker in text
            for marker in ("اصطناعي", "ليس تشريعاً حقيقياً", "ليس قانوناً سعودياً", "ليس نصيحة قانونية")
        ):
            raise ValueError("every demo passage must carry synthetic legal-status disclaimers")
        chunks[chunk_id] = RetrievalChunk(
            chunk_id=chunk_id,
            document_id=_required(row, "document_id"),
            source_id=_required(row, "source_id"),
            unit_type="demo_provision",
            display_text=text,
            search_text=_required(row, "search_text"),
            source_unit_ids=(f"{chunk_id}-unit",),
            chunk_policy_hash="0" * 64,
            normalization_policy_id="arabic-raw-v1",
            normalization_policy_hash="0" * 64,
            token_count=max(1, len(text.split())),
        )
        metadata[chunk_id] = {
            "document_title": title,
            "article": cast(str | None, row.get("article")),
            "page": cast(str | None, row.get("page")),
            "source_url": None,
        }
    if not chunks:
        raise ValueError("demo corpus is empty")
    return chunks, metadata, raw_bytes


def _required(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"demo chunk {key} is invalid")
    return value


__all__ = ["DEFAULT_DEMO_ROOT", "MODEL_ID", "MODEL_REVISION", "DemoCorpus", "load_demo_corpus"]
