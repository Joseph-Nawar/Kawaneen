"""Private context-pack and tracked text-free artifact helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from kawaneen.grounding.contracts import ContextPack, TokenCounter

PRIVATE_ROOT = Path("artifacts/private/phase9_grounding")
TRACKED_ROOT = Path("data/manifests/grounding")
EVALUATION_ROOT = Path("data/evaluation")


def context_pack_fingerprint(
    pack: ContextPack,
    *,
    phase8_selection_sha256: str,
    query_id: str,
    canonical_corpus_hash: str,
    assembly_policy_version: str,
    token_counter: TokenCounter,
    max_context_tokens: int,
) -> str:
    payload = {
        "phase8_selection_sha256": phase8_selection_sha256,
        "query_id": query_id,
        "ordered_input_chunk_ids": list(pack.input_chunk_ids),
        "canonical_corpus_hash": canonical_corpus_hash,
        "chunk_policy_hash": pack.chunk_policy_hash,
        "assembly_policy_version": assembly_policy_version,
        "token_counter_identity": token_counter.identity,
        "max_context_tokens": max_context_tokens,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_private_pack(path: Path, pack: ContextPack, fingerprint: str) -> None:
    payload = pack.model_dump(mode="json")
    payload["fingerprint"] = fingerprint
    _write_json(path, payload)


def write_tracked_json(path: Path, payload: object) -> None:
    if _contains_source_text(payload):
        raise ValueError(f"tracked grounding artifact contains source text: {path}")
    _write_json(path, payload)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _contains_source_text(value: object, *, key: str = "") -> bool:
    lowered = key.lower()
    if lowered in {"display_text", "search_text", "quoted_text", "answer_text", "gold_answer"}:
        return True
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return any(
            _contains_source_text(child, key=str(name)) for name, child in mapping.items()
        )
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return any(_contains_source_text(child, key=key) for child in sequence)
    return False
