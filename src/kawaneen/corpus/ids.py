"""Stable UUIDv5 identifiers for canonical corpus records."""

from __future__ import annotations

from uuid import NAMESPACE_URL, UUID, uuid5

_NAMESPACE: UUID = uuid5(NAMESPACE_URL, "https://kawaneen.local/canonical-corpus/v1")


def canonical_id(
    source_id: str,
    source_version: str,
    source_path: str,
    source_row: int,
    source_field: str,
    kind: str = "unit",
) -> str:
    """Return a deterministic identifier based on source location, never source text."""

    name = "|".join((kind, source_id, source_version, source_path, str(source_row), source_field))
    return str(uuid5(_NAMESPACE, name))
