"""Deterministic Parquet output for ignored canonical corpus views."""

from __future__ import annotations

import hashlib
import json
import os
import re
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pyarrow as _pyarrow
import pyarrow.parquet as _parquet

from kawaneen.corpus.models import RawAccounting

pa: Any = cast(Any, _pyarrow)
pq: Any = cast(Any, _parquet)

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def canonical_root(root: Path, source_id: str, version: str) -> Path:
    if root.is_absolute() and root == Path("/"):
        raise ValueError("canonical output root is unsafe")
    parts = root.parts
    if any(left == "data" and right == "raw" for left, right in pairwise(parts)):
        raise ValueError("canonical outputs cannot be written under data/raw")
    if not _SAFE.fullmatch(source_id) or not _SAFE.fullmatch(version):
        raise ValueError("canonical source and version must be safe path components")
    return root / source_id / version


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(records: list[dict[str, Any]], path: Path, schema: Any) -> dict[str, Any]:
    """Write sorted records atomically and return sanitized file metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=schema)
    partial = path.with_name(f"{path.name}.partial")
    try:
        pq.write_table(table, partial, compression="zstd", use_dictionary=False)
        os.replace(partial, path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return {"path": path.as_posix(), "size": path.stat().st_size, "sha256": _hash(path)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    try:
        partial.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def documents_schema() -> Any:
    return pa.schema(
        [
            ("document_id", pa.string()),
            ("kind", pa.string()),
            ("title", pa.string()),
            ("source_id", pa.string()),
            ("source_version", pa.string()),
            ("source_path", pa.string()),
            ("source_row", pa.int64()),
            ("source_field", pa.string()),
            ("split", pa.string()),
            ("raw_article_label", pa.string()),
            ("derived_article_ordinal", pa.int64()),
            ("reconstruction_status", pa.string()),
            ("source_metadata_json", pa.string()),
        ]
    )


def units_schema() -> Any:
    return pa.schema(
        [
            ("unit_id", pa.string()),
            ("document_id", pa.string()),
            ("unit_type", pa.string()),
            ("text", pa.string()),
            ("source_id", pa.string()),
            ("source_version", pa.string()),
            ("source_path", pa.string()),
            ("source_row", pa.int64()),
            ("source_field", pa.string()),
            ("split", pa.string()),
            ("ordinal", pa.int64()),
        ]
    )


def fragments_schema() -> Any:
    return pa.schema(
        [
            ("fragment_id", pa.string()),
            ("source_id", pa.string()),
            ("source_version", pa.string()),
            ("source_path", pa.string()),
            ("source_row", pa.int64()),
            ("source_field", pa.string()),
            ("raw_label", pa.string()),
            ("law_name", pa.string()),
            ("law_type", pa.string()),
            ("derived_article_ordinal", pa.int64()),
            ("explicit_part", pa.int64()),
            ("article_label_structural_key", pa.string()),
            ("article_parse_confidence", pa.string()),
            ("article_status_marker", pa.string()),
            ("part_index", pa.int64()),
            ("text", pa.string()),
        ]
    )


def reconstruction_schema() -> Any:
    return pa.schema(
        [
            ("law_name", pa.string()),
            ("raw_article_label", pa.string()),
            ("status", pa.string()),
            ("article_label_structural_key", pa.string()),
            ("article_ordinal", pa.int64()),
            ("article_parse_confidence", pa.string()),
            ("part_index", pa.int64()),
            ("article_status_marker", pa.string()),
            ("fragment_ids_json", pa.string()),
            ("operations_json", pa.string()),
        ]
    )


def accounting_metadata(accounting: RawAccounting) -> dict[str, Any]:
    return accounting.model_dump()
