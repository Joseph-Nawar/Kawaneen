from __future__ import annotations

from pathlib import Path

import pytest

from kawaneen.acquisition.storage import StorageError, copy_immutable, source_root


def test_source_root_is_namespaced_and_relative(tmp_path: Path) -> None:
    assert source_root(tmp_path, "alarb", "rev") == tmp_path / "alarb" / "rev"
    with pytest.raises(StorageError):
        source_root(tmp_path, "../escape", "rev")
    with pytest.raises(StorageError):
        source_root(tmp_path, "alarb", "/absolute")


def test_copy_immutable_hashes_and_cleans_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable bytes")
    destination = tmp_path / "raw"
    result = copy_immutable(source, destination, "file.bin")
    assert result.size == len(b"stable bytes")
    assert result.sha256
    assert (destination / "file.bin").read_bytes() == b"stable bytes"
    assert not list(destination.rglob("*.partial"))


def test_copy_rejects_traversal_and_existing_different_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"one")
    destination = tmp_path / "raw"
    copy_immutable(source, destination, "file.bin")
    with pytest.raises(StorageError, match="relative"):
        copy_immutable(source, destination, "../escape.bin")
    source.write_bytes(b"two")
    with pytest.raises(StorageError, match="immutable"):
        copy_immutable(source, destination, "file.bin")


def test_copy_rejects_symlink_escape(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"one")
    destination = tmp_path / "raw"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination.mkdir()
    (destination / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(StorageError, match="escapes"):
        copy_immutable(source, destination, "nested/file.bin")
