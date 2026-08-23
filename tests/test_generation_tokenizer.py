from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kawaneen.generation.tokenizer as tokenizer_module
from kawaneen.generation.tokenizer import LazyHuggingFaceTokenizer


def test_tokenizer_preflight_loads_once_and_reuses_instance() -> None:
    calls: list[tuple[str, str]] = []

    def loader(identity: str, revision: str) -> object:
        calls.append((identity, revision))
        return lambda text: {"input_ids": list(text)}

    tokenizer = LazyHuggingFaceTokenizer(
        identity="Qwen/Qwen3-4B-Instruct-2507",
        revision="a" * 40,
        loader=loader,
    )

    tokenizer.preflight()
    assert tokenizer.count("one") == 3
    tokenizer.preflight()
    assert calls == [("Qwen/Qwen3-4B-Instruct-2507", "a" * 40)]


def test_default_loader_uses_exact_local_snapshot_and_never_network(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[dict[str, object]] = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            calls.append({"args": args, "kwargs": kwargs})
            return lambda _text: {"input_ids": [1]}

    monkeypatch.setattr(
        tokenizer_module, "_resolve_local_tokenizer_snapshot", lambda *_: snapshot
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=AutoTokenizer),
    )

    tokenizer_module._default_loader(
        "Qwen/Qwen3-4B-Instruct-2507",
        "a" * 40,
    )

    assert calls == [
        {
            "args": (str(snapshot),),
            "kwargs": {"revision": "a" * 40, "local_files_only": True},
        }
    ]


def test_missing_exact_local_tokenizer_fails_preflight(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(tokenizer_module, "_resolve_local_tokenizer_snapshot", lambda *_: None)

    with pytest.raises(ValueError, match="exact pinned tokenizer revision is unavailable locally"):
        tokenizer_module._default_loader(
            "Qwen/Qwen3-4B-Instruct-2507",
            "a" * 40,
        )
