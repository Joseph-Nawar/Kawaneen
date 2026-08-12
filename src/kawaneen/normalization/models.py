"""Typed immutable models for Arabic normalization experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class NormalizationPolicy:
    """Versioned, hash-addressable normalization policy."""

    policy_id: str
    version: int
    transforms: tuple[str, ...]
    config: Mapping[str, object]
    policy_hash: str


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Normalized text plus sanitized transform counts."""

    search_text: str
    transform_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform_counts", MappingProxyType(dict(self.transform_counts)))
