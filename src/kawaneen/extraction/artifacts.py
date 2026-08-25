"""Atomic private and text-free tracked JSON artifact helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _assert_text_free(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for key, child in mapping.items():
            label = str(key).casefold()
            if isinstance(child, str) and any(
                marker in label for marker in ("text", "quote", "prompt", "answer")
            ):
                raise ValueError(f"source text is not allowed in tracked artifact: {path}.{key}")
            _assert_text_free(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(cast(Sequence[object], value)):
            _assert_text_free(child, f"{path}[{index}]")


def write_private_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write(path, payload)


def write_text_free_json(path: Path, payload: Mapping[str, object]) -> None:
    _assert_text_free(payload)
    _atomic_write(path, payload)
