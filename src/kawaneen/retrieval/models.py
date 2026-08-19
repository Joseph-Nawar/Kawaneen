"""Typed immutable records shared by retrieval baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from kawaneen.evaluation.models import DatasetItem


@dataclass(frozen=True, slots=True)
class RetrievalChunk:
    chunk_id: str
    document_id: str
    source_id: str
    unit_type: str
    display_text: str
    search_text: str
    source_unit_ids: tuple[str, ...]
    chunk_policy_hash: str
    normalization_policy_id: str
    normalization_policy_hash: str
    token_count: int
    source_spans: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalRelease:
    items: tuple[DatasetItem, ...]
    chunks: tuple[RetrievalChunk, ...]
    phase6_manifest: Mapping[str, object]
    corpus_manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase6_manifest", MappingProxyType(dict(self.phase6_manifest)))
        object.__setattr__(self, "corpus_manifest", MappingProxyType(dict(self.corpus_manifest)))

    def split_items(self, split: str, *, allow_holdout: bool = False) -> tuple[DatasetItem, ...]:
        if split == "holdout" and not allow_holdout:
            raise PermissionError("holdout access requires allow_holdout=True")
        return tuple(item for item in self.items if item.split.value == split)
