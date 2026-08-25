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

    monkeypatch.setattr(tokenizer_module, "_resolve_local_tokenizer_snapshot", lambda *_: snapshot)
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


def test_codepoint_tokenizer_and_revision_validation() -> None:
    codepoint = tokenizer_module.CodepointTokenizer()
    assert codepoint.fingerprint.identity == "codepoint-v1"
    assert codepoint.count("قانون") == 5

    with pytest.raises(ValueError, match="full 40-character SHA"):
        LazyHuggingFaceTokenizer(identity="test", revision="short", loader=lambda *_: object())


@pytest.mark.parametrize(
    "output",
    [lambda _text: object(), lambda _text: {"input_ids": (1, 2)}, lambda _text: {"other": []}],
)
def test_tokenizer_rejects_malformed_loader_output(output) -> None:  # type: ignore[no-untyped-def]
    tokenizer = LazyHuggingFaceTokenizer(
        identity="test", revision="a" * 40, loader=lambda *_: output
    )

    with pytest.raises(TypeError, match="input_ids"):
        tokenizer.count("text")


def test_local_tokenizer_snapshot_requires_all_pinned_files(monkeypatch, tmp_path: Path) -> None:
    revision = "b" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    files = {
        "tokenizer.json": snapshot / "tokenizer.json",
        "tokenizer_config.json": snapshot / "tokenizer_config.json",
    }

    def cached(_identity: str, *, filename: str, revision: str) -> str | None:
        return str(files[filename]) if revision == revision else None

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=cached),
    )

    assert tokenizer_module._resolve_local_tokenizer_snapshot("test", revision) == snapshot


def test_local_tokenizer_snapshot_rejects_incomplete_or_wrong_cache(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "c" * 40
    wrong = tmp_path / "wrong"
    wrong.mkdir()

    def missing_config(_identity: str, *, filename: str, revision: str) -> str | None:
        return str(wrong / filename) if filename == "tokenizer.json" else None

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=missing_config),
    )

    assert tokenizer_module._resolve_local_tokenizer_snapshot("test", revision) is None
