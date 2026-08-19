# pyright: basic
"""Private dense embedding caches, including resumable checkpoint blocks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BLOCK_SIZE = 1024
NORMALIZED_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class CheckpointEncodingResult:
    vectors: np.ndarray
    batch_size: int
    cache_status: str


def embedding_cache_fingerprint(
    *,
    corpus_hash: str,
    policy_hash: str,
    normalization_policy_hash: str,
    model_id: str,
    model_revision: str,
    formatting_contract: str,
    max_length: int,
    embedding_dimension: int,
    normalize: bool,
    dtype: str,
) -> str:
    payload = {
        "corpus_hash": corpus_hash,
        "policy_hash": policy_hash,
        "normalization_policy_hash": normalization_policy_hash,
        "model_id": model_id,
        "model_revision": model_revision,
        "formatting_contract": formatting_contract,
        "max_length": max_length,
        "embedding_dimension": embedding_dimension,
        "normalize": normalize,
        "dtype": dtype,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ordered_ids_hash(chunk_ids: Sequence[str]) -> str:
    return _sha256_bytes(json.dumps(list(chunk_ids), separators=(",", ":")).encode())


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )


def _atomic_write_npy(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.save(handle, vectors, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_cached_embeddings(
    path: Path, vectors: np.ndarray, chunk_ids: Sequence[str], *, fingerprint: str
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    values = np.asarray(vectors)
    if values.ndim != 2 or values.shape[0] != len(chunk_ids):
        raise ValueError("cached vectors and chunk IDs must have matching rows")
    if values.dtype != np.dtype("float32"):
        raise ValueError("cached vectors must have float32 dtype")
    _atomic_write_npy(path / "vectors.npy", values)
    _atomic_write_json(path / "ids.json", list(chunk_ids))
    _atomic_write_json(
        path / "metadata.json",
        {"fingerprint": fingerprint, "dtype": "float32", "dimension": values.shape[1]},
    )


def load_cached_embeddings(path: Path, *, fingerprint: str) -> tuple[np.ndarray, tuple[str, ...]]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("embedding cache fingerprint mismatch")
    if metadata.get("dtype") != "float32":
        raise ValueError("embedding cache dtype metadata mismatch")
    vectors = np.load(path / "vectors.npy", allow_pickle=False)
    ids = tuple(str(value) for value in json.loads((path / "ids.json").read_text(encoding="utf-8")))
    if (
        vectors.ndim != 2
        or vectors.dtype != np.dtype("float32")
        or vectors.shape[0] != len(ids)
        or vectors.shape[1] != int(metadata["dimension"])
    ):
        raise ValueError("embedding cache shape metadata mismatch")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("embedding cache contains NaN or Inf")
    return np.asarray(vectors, dtype=np.float32), ids


def _block_entry(manifest: Mapping[str, Any], block_index: int) -> Mapping[str, Any]:
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list) or block_index < 0 or block_index >= len(blocks):
        raise ValueError("checkpoint manifest block table is invalid")
    entry = blocks[block_index]
    if not isinstance(entry, Mapping) or int(entry.get("block_index", -1)) != block_index:
        raise ValueError("checkpoint manifest block index is invalid")
    return entry


def _manifest_for(
    *,
    fingerprint: str,
    chunk_ids: Sequence[str],
    embedding_dimension: int,
    block_size: int,
    model_config: Mapping[str, object] | None,
) -> dict[str, Any]:
    total_chunks = len(chunk_ids)
    total_blocks = (total_chunks + block_size - 1) // block_size
    return {
        "schema_version": 2,
        "status": "dense_checkpoint_encoding",
        "fingerprint": fingerprint,
        "total_chunks": total_chunks,
        "total_blocks": total_blocks,
        "block_size": block_size,
        "embedding_dimension": embedding_dimension,
        "dtype": "float32",
        "normalize": True,
        "chunk_ids_hash": _ordered_ids_hash(chunk_ids),
        "model_config": dict(model_config or {}),
        "blocks": [
            {
                "block_index": block_index,
                "start": start,
                "end": min(start + block_size, total_chunks),
                "row_count": min(start + block_size, total_chunks) - start,
                "dimension": embedding_dimension,
                "dtype": "float32",
                "sha256": "",
                "chunk_ids_hash": _ordered_ids_hash(
                    chunk_ids[start : min(start + block_size, total_chunks)]
                ),
            }
            for block_index, start in enumerate(range(0, total_chunks, block_size))
        ],
    }


def _initial_progress(manifest: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "in_progress",
        "fingerprint": fingerprint,
        "total_blocks": int(manifest["total_blocks"]),
        "total_chunks": int(manifest["total_chunks"]),
        "completed_blocks": 0,
        "completed_chunks": 0,
        "elapsed_compute_seconds": 0.0,
    }


def prepare_checkpoint_cache(
    path: Path,
    *,
    fingerprint: str,
    chunk_ids: Sequence[str],
    embedding_dimension: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    model_config: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if block_size < 1:
        raise ValueError("checkpoint block size must be positive")
    if embedding_dimension < 1:
        raise ValueError("embedding dimension must be positive")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("checkpoint chunk IDs must be unique")
    path.mkdir(parents=True, exist_ok=True)
    (path / "blocks").mkdir(exist_ok=True)
    (path / "chunk_ids").mkdir(exist_ok=True)
    manifest_path = path / "manifest.json"
    expected = _manifest_for(
        fingerprint=fingerprint,
        chunk_ids=chunk_ids,
        embedding_dimension=embedding_dimension,
        block_size=block_size,
        model_config=model_config,
    )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "schema_version",
            "fingerprint",
            "total_chunks",
            "total_blocks",
            "block_size",
            "embedding_dimension",
            "dtype",
            "normalize",
            "chunk_ids_hash",
        ):
            if manifest.get(key) != expected[key]:
                raise ValueError(f"checkpoint cache {key} mismatch")
        if manifest.get("model_config", {}) != expected["model_config"]:
            raise ValueError("checkpoint cache model config mismatch")
        if not isinstance(manifest.get("blocks"), list) or len(manifest["blocks"]) != len(
            expected["blocks"]
        ):
            raise ValueError("checkpoint cache block count mismatch")
        for actual, expected_entry in zip(manifest["blocks"], expected["blocks"], strict=True):
            if not isinstance(actual, Mapping):
                raise ValueError("checkpoint cache block entry is invalid")
            if any(
                actual.get(key) != expected_entry[key]
                for key in (
                    "block_index",
                    "start",
                    "end",
                    "row_count",
                    "dimension",
                    "dtype",
                    "chunk_ids_hash",
                )
            ):
                raise ValueError("checkpoint cache block contract mismatch")
    else:
        manifest = expected
        _atomic_write_json(manifest_path, manifest)
    progress_path = path / "progress.json"
    if not progress_path.is_file():
        _atomic_write_json(progress_path, _initial_progress(manifest, fingerprint))
    return manifest


def _valid_norms(vectors: np.ndarray) -> bool:
    if vectors.ndim != 2 or vectors.dtype != np.dtype("float32"):
        return False
    if not np.all(np.isfinite(vectors)):
        return False
    norms = np.linalg.norm(vectors, axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= NORMALIZED_NORM_TOLERANCE))


def validate_checkpoint_block(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    block_index: int,
    expected_chunk_ids: Sequence[str],
) -> bool:
    try:
        if not validate_checkpoint_block_manifest(path, manifest=manifest, block_index=block_index):
            return False
        entry = _block_entry(manifest, block_index)
        start = int(entry["start"])
        end = int(entry["end"])
        ids_path = path / "chunk_ids" / f"block_{block_index:05d}.json"
        ids = tuple(str(value) for value in json.loads(ids_path.read_text(encoding="utf-8")))
        if ids != tuple(expected_chunk_ids):
            return False
        return len(ids) == end - start
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def validate_checkpoint_block_manifest(
    path: Path, *, manifest: Mapping[str, Any], block_index: int
) -> bool:
    """Validate a block using only checkpoint metadata and block-local files."""
    try:
        entry = _block_entry(manifest, block_index)
        start = int(entry["start"])
        end = int(entry["end"])
        expected_start = block_index * int(manifest["block_size"])
        expected_end = min(
            expected_start + int(manifest["block_size"]), int(manifest["total_chunks"])
        )
        if (
            start != expected_start
            or end != expected_end
            or end <= start
            or int(entry["row_count"]) != end - start
        ):
            return False
        vectors_path = path / "blocks" / f"block_{block_index:05d}.npy"
        ids_path = path / "chunk_ids" / f"block_{block_index:05d}.json"
        if not vectors_path.is_file() or not ids_path.is_file() or not entry.get("sha256"):
            return False
        if _sha256_bytes(vectors_path.read_bytes()) != entry["sha256"]:
            return False
        ids = tuple(str(value) for value in json.loads(ids_path.read_text(encoding="utf-8")))
        if len(ids) != end - start or _ordered_ids_hash(ids) != entry.get("chunk_ids_hash"):
            return False
        vectors = np.load(vectors_path, allow_pickle=False)
        return not (
            vectors.shape != (int(entry["row_count"]), int(entry["dimension"]))
            or vectors.dtype != np.dtype("float32")
            or not _valid_norms(vectors)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _valid_block_indices(
    path: Path, manifest: Mapping[str, Any], chunk_ids: Sequence[str]
) -> tuple[int, ...]:
    return tuple(
        block_index
        for block_index in range(int(manifest["total_blocks"]))
        if validate_checkpoint_block(
            path,
            manifest=manifest,
            block_index=block_index,
            expected_chunk_ids=chunk_ids[
                int(manifest["blocks"][block_index]["start"]) : int(
                    manifest["blocks"][block_index]["end"]
                )
            ],
        )
    )


def _write_progress(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    fingerprint: str,
    completed_blocks: int,
    completed_chunks: int,
    elapsed_compute_seconds: float,
    complete: bool,
) -> None:
    _atomic_write_json(
        path / "progress.json",
        {
            "schema_version": 1,
            "status": "complete" if complete else "in_progress",
            "fingerprint": fingerprint,
            "total_blocks": int(manifest["total_blocks"]),
            "total_chunks": int(manifest["total_chunks"]),
            "completed_blocks": completed_blocks,
            "completed_chunks": completed_chunks,
            "elapsed_compute_seconds": elapsed_compute_seconds,
        },
    )


def _save_checkpoint_block(
    path: Path,
    manifest: dict[str, Any],
    block_index: int,
    vectors: np.ndarray,
    chunk_ids: Sequence[str],
) -> None:
    entry = _block_entry(manifest, block_index)
    expected_rows = int(entry["row_count"])
    values = np.asarray(vectors)
    if (
        values.shape != (expected_rows, int(entry["dimension"]))
        or values.dtype != np.dtype("float32")
        or not _valid_norms(values)
    ):
        raise ValueError("dense checkpoint block does not satisfy the embedding contract")
    if len(chunk_ids) != expected_rows:
        raise ValueError("dense checkpoint block has the wrong chunk ID count")
    vectors_path = path / "blocks" / f"block_{block_index:05d}.npy"
    ids_path = path / "chunk_ids" / f"block_{block_index:05d}.json"
    _atomic_write_npy(vectors_path, values)
    _atomic_write_json(ids_path, list(chunk_ids))
    updated_entry = dict(entry)
    updated_entry["sha256"] = _sha256_bytes(vectors_path.read_bytes())
    blocks = list(manifest["blocks"])
    blocks[block_index] = updated_entry
    manifest["blocks"] = blocks
    _atomic_write_json(path / "manifest.json", manifest)


def checkpoint_cache_status(
    path: Path, *, chunk_ids: Sequence[str], fingerprint: str
) -> dict[str, object]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError("checkpoint cache fingerprint mismatch")
    if manifest.get("chunk_ids_hash") != _ordered_ids_hash(chunk_ids):
        raise ValueError("checkpoint cache chunk ID order mismatch")
    completed = _valid_block_indices(path, manifest, chunk_ids)
    total_chunks = len(chunk_ids)
    completed_chunks = sum(int(manifest["blocks"][index]["row_count"]) for index in completed)
    progress = json.loads((path / "progress.json").read_text(encoding="utf-8"))
    elapsed = float(progress.get("elapsed_compute_seconds", 0.0))
    return {
        "completed_blocks": len(completed),
        "total_blocks": int(manifest["total_blocks"]),
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "percentage": (100.0 * completed_chunks / total_chunks) if total_chunks else 100.0,
        "elapsed_recorded_compute_seconds": elapsed,
        "estimated_remaining_blocks": int(manifest["total_blocks"]) - len(completed),
        "cache_fingerprint": fingerprint,
    }


def checkpoint_cache_status_from_manifest(path: Path, *, fingerprint: str) -> dict[str, object]:
    """Report checkpoint progress without loading canonical corpus text or IDs."""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError("checkpoint cache fingerprint mismatch")
    total_blocks = int(manifest["total_blocks"])
    total_chunks = int(manifest["total_chunks"])
    completed = tuple(
        block_index
        for block_index in range(total_blocks)
        if validate_checkpoint_block_manifest(path, manifest=manifest, block_index=block_index)
    )
    completed_chunks = sum(int(manifest["blocks"][index]["row_count"]) for index in completed)
    progress = json.loads((path / "progress.json").read_text(encoding="utf-8"))
    if progress.get("fingerprint") != fingerprint:
        raise ValueError("checkpoint progress fingerprint mismatch")
    return {
        "completed_blocks": len(completed),
        "total_blocks": total_blocks,
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "percentage": (100.0 * completed_chunks / total_chunks) if total_chunks else 100.0,
        "elapsed_recorded_compute_seconds": float(progress.get("elapsed_compute_seconds", 0.0)),
        "estimated_remaining_blocks": total_blocks - len(completed),
        "cache_fingerprint": fingerprint,
    }


def encode_corpus_checkpointed(
    texts: Sequence[str],
    chunk_ids: Sequence[str],
    path: Path,
    *,
    fingerprint: str,
    encoder: Callable[[tuple[str, ...], int], object],
    embedding_dimension: int,
    batch_size: int = 1,
    block_size: int = DEFAULT_BLOCK_SIZE,
    model_config: Mapping[str, object] | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> CheckpointEncodingResult:
    if len(texts) != len(chunk_ids):
        raise ValueError("dense texts and chunk IDs must have matching rows")
    if batch_size < 1:
        raise ValueError("dense batch size must be positive")
    if (path / "manifest.json").is_file() is False and (path / "metadata.json").is_file():
        vectors, _ = load_cached_embeddings(path, fingerprint=fingerprint)
        cached_ids = tuple(
            str(value) for value in json.loads((path / "ids.json").read_text(encoding="utf-8"))
        )
        if tuple(chunk_ids) != cached_ids:
            raise ValueError("legacy embedding cache chunk ID order mismatch")
        return CheckpointEncodingResult(vectors, batch_size, "legacy_hit")

    manifest = prepare_checkpoint_cache(
        path,
        fingerprint=fingerprint,
        chunk_ids=chunk_ids,
        embedding_dimension=embedding_dimension,
        block_size=block_size,
        model_config=model_config,
    )
    progress = json.loads((path / "progress.json").read_text(encoding="utf-8"))
    elapsed_compute_seconds = float(progress.get("elapsed_compute_seconds", 0.0))
    resolved_batch_size = batch_size
    for block_index in range(int(manifest["total_blocks"])):
        start = int(manifest["blocks"][block_index]["start"])
        end = int(manifest["blocks"][block_index]["end"])
        expected_ids = chunk_ids[start:end]
        if validate_checkpoint_block(
            path,
            manifest=manifest,
            block_index=block_index,
            expected_chunk_ids=expected_ids,
        ):
            if progress_callback is not None:
                progress_callback(
                    {
                        "block_index": block_index,
                        "completed_blocks": len(_valid_block_indices(path, manifest, chunk_ids)),
                        "total_blocks": int(manifest["total_blocks"]),
                    }
                )
            continue
        current_batch_size = resolved_batch_size
        started = time.perf_counter()
        while True:
            try:
                raw = encoder(tuple(texts[start:end]), current_batch_size)
                vectors = np.asarray(raw, dtype=np.float32)
                break
            except RuntimeError as exc:
                if current_batch_size <= 1 or "out of memory" not in str(exc).lower():
                    raise
                current_batch_size //= 2
        elapsed_compute_seconds += time.perf_counter() - started
        resolved_batch_size = current_batch_size
        _save_checkpoint_block(path, manifest, block_index, vectors, expected_ids)
        completed = _valid_block_indices(path, manifest, chunk_ids)
        completed_chunks = sum(int(manifest["blocks"][index]["row_count"]) for index in completed)
        _write_progress(
            path,
            manifest=manifest,
            fingerprint=fingerprint,
            completed_blocks=len(completed),
            completed_chunks=completed_chunks,
            elapsed_compute_seconds=elapsed_compute_seconds,
            complete=False,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "block_index": block_index,
                    "completed_blocks": len(completed),
                    "total_blocks": int(manifest["total_blocks"]),
                }
            )

    completed = _valid_block_indices(path, manifest, chunk_ids)
    if len(completed) != int(manifest["total_blocks"]):
        raise ValueError("dense checkpoint consolidation requires every block")
    blocks = [
        np.load(path / "blocks" / f"block_{index:05d}.npy", allow_pickle=False)
        for index in range(int(manifest["total_blocks"]))
    ]
    consolidated = (
        np.concatenate(blocks, axis=0)
        if blocks
        else np.empty((0, embedding_dimension), dtype=np.float32)
    )
    save_cached_embeddings(
        path, consolidated.astype(np.float32, copy=False), chunk_ids, fingerprint=fingerprint
    )
    _write_progress(
        path,
        manifest=manifest,
        fingerprint=fingerprint,
        completed_blocks=len(completed),
        completed_chunks=len(chunk_ids),
        elapsed_compute_seconds=elapsed_compute_seconds,
        complete=True,
    )
    return CheckpointEncodingResult(
        consolidated.astype(np.float32, copy=False), resolved_batch_size, "complete"
    )
