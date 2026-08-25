"""Exact-tokenizer adapters with lazy optional Transformers loading."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, cast

from kawaneen.generation.contracts import TokenizerFingerprint


class TokenizerAdapter(Protocol):
    @property
    def fingerprint(self) -> TokenizerFingerprint: ...

    def count(self, text: str) -> int: ...


class CodepointTokenizer:
    """Deterministic test-only counter; not the real generation tokenizer."""

    @property
    def fingerprint(self) -> TokenizerFingerprint:
        return TokenizerFingerprint(identity="codepoint-v1")

    def count(self, text: str) -> int:
        return len(text)


class LazyHuggingFaceTokenizer:
    def __init__(
        self,
        *,
        identity: str,
        revision: str,
        loader: Callable[[str, str], object] | None = None,
    ) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("tokenizer revision must be a full 40-character SHA")
        self.identity = identity
        self.revision = revision
        self.loader = loader or _default_loader
        self._tokenizer: object | None = None

    @property
    def fingerprint(self) -> TokenizerFingerprint:
        return TokenizerFingerprint(identity=self.identity, revision=self.revision)

    def preflight(self) -> None:
        """Load the exact tokenizer once before any query processing."""

        self._ensure_loaded()

    def count(self, text: str) -> int:
        value = self._ensure_loaded()
        if not callable(value):
            raise TypeError("tokenizer loader did not return a callable tokenizer")
        encoded_value = value(text)
        if not isinstance(encoded_value, Mapping):
            raise TypeError("tokenizer output has no input_ids list")
        encoded = cast(Mapping[str, object], encoded_value)
        ids = encoded.get("input_ids")
        if not isinstance(ids, list):
            raise TypeError("tokenizer output has no input_ids list")
        return len(cast(list[object], ids))

    def _ensure_loaded(self) -> object:
        if self._tokenizer is None:
            self._tokenizer = self.loader(self.identity, self.revision)
        return self._tokenizer


def _resolve_local_tokenizer_snapshot(identity: str, revision: str) -> Path | None:
    """Resolve a pinned tokenizer snapshot using cache metadata only."""

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as error:  # pragma: no cover - dependency is production-locked
        raise ValueError("huggingface_hub is required for local tokenizer preflight") from error
    tokenizer_path = try_to_load_from_cache(
        identity,
        filename="tokenizer.json",
        revision=revision,
    )
    if not isinstance(tokenizer_path, str):
        return None
    snapshot = Path(tokenizer_path).parent
    if snapshot.name != revision:
        return None
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        cached = try_to_load_from_cache(identity, filename=filename, revision=revision)
        if not isinstance(cached, str) or Path(cached).parent != snapshot:
            return None
    return snapshot


def _default_loader(identity: str, revision: str) -> object:
    snapshot = _resolve_local_tokenizer_snapshot(identity, revision)
    if snapshot is None:
        raise ValueError(
            f"exact pinned tokenizer revision is unavailable locally: {identity}@{revision}"
        )
    from transformers import AutoTokenizer

    return cast(
        object,
        AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            str(snapshot),
            revision=revision,
            local_files_only=True,
        ),
    )
