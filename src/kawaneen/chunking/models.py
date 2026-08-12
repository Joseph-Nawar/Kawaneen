"""Immutable typed models for Phase 5 structure and retrieval chunks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class SourceSpan:
    unit_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("source span requires a unit ID")
        if self.start < 0 or self.end < self.start:
            raise ValueError("source span bounds are invalid")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class CitationAnchor:
    kind: str
    label: str = ""
    source_unit_id: str | None = None


@dataclass(frozen=True, slots=True)
class StructureNode:
    node_id: str
    kind: str
    document_id: str
    parent_id: str | None
    source_unit_id: str | None
    spans: tuple[SourceSpan, ...]
    structure_path: tuple[str, ...]
    children: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    policy_id: str
    version: int
    config: Mapping[str, object]
    policy_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))


@dataclass(frozen=True, slots=True)
class LegalChunk:
    chunk_id: str
    strategy_id: str
    chunk_policy_hash: str
    source_unit_ids: tuple[str, ...]
    display_text: str
    search_text: str
    source_spans: tuple[SourceSpan, ...]
    parent_id: str | None
    ancestor_ids: tuple[str, ...]
    sibling_ids: tuple[str, ...]
    structure_path: tuple[str, ...]
    citation_anchor: CitationAnchor | None
    token_count: int
    normalization_policy_id: str
    normalization_policy_hash: str
    provenance: Mapping[str, object]
    context_source_spans: tuple[SourceSpan, ...] = ()
    indexed_child_ids: tuple[str, ...] = ()
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("chunk token count cannot be negative")
        if not self.source_spans:
            raise ValueError("chunk requires at least one source span")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def deterministic_chunk_id(
    strategy_id: str, source_identity: str, spans: Sequence[SourceSpan], policy_hash: str = ""
) -> str:
    """Hash source identity/spans/policy, never a positional chunk number."""

    payload = {
        "policy_hash": policy_hash,
        "source_identity": source_identity,
        "spans": [
            span.__dict__
            if hasattr(span, "__dict__")
            else {
                "unit_id": span.unit_id,
                "start": span.start,
                "end": span.end,
            }
            for span in spans
        ],
        "strategy_id": strategy_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{strategy_id}:{digest[:32]}"
