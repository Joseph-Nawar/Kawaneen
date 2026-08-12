"""Versioned Phase 5 chunk policies."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import cast

from kawaneen.chunking.models import ChunkPolicy

_CONFIGS: tuple[dict[str, object], ...] = (
    {
        "policy_id": "fixed-256-v1",
        "version": 1,
        "strategy": "fixed_window",
        "token_target": 256,
        "token_maximum": 256,
        "overlap": 32,
    },
    {
        "policy_id": "fixed-512-v1",
        "version": 1,
        "strategy": "fixed_window",
        "token_target": 512,
        "token_maximum": 512,
        "overlap": 64,
    },
    {
        "policy_id": "legal-structure-v1",
        "version": 1,
        "strategy": "legal_structure",
        "token_target": 384,
        "token_maximum": 512,
        "overlap": 0,
    },
    {
        "policy_id": "legal-structure-neighbor-v1",
        "version": 1,
        "strategy": "legal_structure_neighbor",
        "token_target": 384,
        "token_maximum": 512,
        "overlap": 0,
        "neighbor_scope": "same_parent_previous_current_next",
    },
    {
        "policy_id": "legal-parent-child-v1",
        "version": 1,
        "strategy": "legal_parent_child",
        "token_target": 384,
        "token_maximum": 512,
        "overlap": 0,
        "aggregation": "max_child_score",
    },
)


def _hash(config: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_POLICIES = tuple(
    ChunkPolicy(
        policy_id=str(config["policy_id"]),
        version=cast(int, config["version"]),
        config=MappingProxyType(dict(config)),
        policy_hash=_hash(config),
    )
    for config in _CONFIGS
)
_BY_ID = {policy.policy_id: policy for policy in _POLICIES}


def all_chunk_policies() -> tuple[ChunkPolicy, ...]:
    return _POLICIES


def get_chunk_policy(policy_id: str) -> ChunkPolicy:
    return _BY_ID[policy_id]


def chunk_policy_configurations() -> tuple[dict[str, object], ...]:
    return tuple(dict(config) for config in _CONFIGS)
