# pyright: basic
"""Atomic per-query reranking checkpoints."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class CheckpointStore:
    def __init__(self, root: Path, *, fingerprint: str) -> None:
        self.root = root
        self.fingerprint = fingerprint
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = root / "manifest.json"
        if not self.manifest_path.is_file():
            _atomic_json(
                self.manifest_path,
                {"schema_version": 1, "fingerprint": fingerprint, "queries": {}},
            )

    def _read_manifest(self) -> dict[str, Any]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != self.fingerprint:
            raise ValueError("checkpoint manifest fingerprint mismatch")
        if not isinstance(payload.get("queries"), dict):
            raise ValueError("checkpoint manifest query table is invalid")
        return payload

    def write(self, query_id: str, payload: Mapping[str, object]) -> None:
        row = {"query_id": query_id, "fingerprint": self.fingerprint, **dict(payload)}
        if "candidate_chunk_ids" not in row:
            row["candidate_chunk_ids"] = list(row.get("ranked_chunk_ids", []))
        query_path = self.root / f"{query_id}.json"
        _atomic_json(query_path, row)
        manifest = self._read_manifest()
        manifest["queries"][query_id] = {
            "path": query_path.name,
            "candidate_chunk_ids": list(
                row.get("candidate_chunk_ids", row.get("ranked_chunk_ids", []))
            ),
            "status": "completed",
        }
        _atomic_json(self.manifest_path, manifest)

    def valid(
        self,
        query_id: str,
        candidate_chunk_ids: Sequence[str],
        *,
        query_fingerprint: str | None = None,
    ) -> bool:
        try:
            manifest = self._read_manifest()
            entry = manifest["queries"].get(query_id)
            if not isinstance(entry, dict) or entry.get("status") != "completed":
                return False
            payload = json.loads((self.root / str(entry["path"])).read_text(encoding="utf-8"))
            if payload.get("fingerprint") != self.fingerprint:
                return False
            if (
                query_fingerprint is not None
                and payload.get("query_fingerprint") != query_fingerprint
            ):
                return False
            if tuple(payload.get("candidate_chunk_ids", ())) != tuple(candidate_chunk_ids):
                return False
            scores = payload.get("scores", ())
            return all(math.isfinite(float(value)) for value in scores)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False


def checkpoint_status(root: Path) -> dict[str, object]:
    """Read only the checkpoint manifest; never opens query result files."""
    path = root / "manifest.json"
    if not path.is_file():
        return {"status": "missing", "total_count": 0, "valid_count": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries", {})
    completed = sum(
        isinstance(value, dict) and value.get("status") == "completed" for value in queries.values()
    )
    return {
        "status": "ready",
        "fingerprint": payload.get("fingerprint", ""),
        "total_count": len(queries),
        "valid_count": completed,
    }
