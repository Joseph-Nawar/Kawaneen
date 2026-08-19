# pyright: basic
"""Atomic Phase-8 artifact helpers and tracked text-free guards."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

_FORBIDDEN_KEYS = {"query_text", "display_text", "search_text", "passage", "gold_answer"}


def is_text_free(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(
            key.lower() not in _FORBIDDEN_KEYS
            and not key.lower().endswith("_text")
            and is_text_free(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(is_text_free(child) for child in value)
    return True


def write_json_atomic(path: Path, payload: object, *, text_free: bool = False) -> None:
    if text_free and not is_text_free(payload):
        raise ValueError(f"tracked artifact must be text-free: {path}")
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
