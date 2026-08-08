"""Secure, byte-preserving local storage for raw source files."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path, PurePosixPath

from kawaneen.acquisition.models import FileDigest


class StorageError(ValueError):
    """Raised when a raw storage safety rule is violated."""


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _component(value: str, label: str) -> str:
    if not value or not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise StorageError(f"{label} is not a safe path component")
    return value


def source_root(raw_root: Path, source_id: str, version: str) -> Path:
    """Return a source/version namespace below raw_root."""

    return raw_root / _component(source_id, "source_id") / _component(version, "version")


def _relative_destination(root: Path, relative_path: str) -> Path:
    candidate = PurePosixPath(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise StorageError("destination path must be relative and cannot escape raw storage")
    destination = root.joinpath(*candidate.parts)
    root_resolved = root.resolve()
    if (
        destination.parent.resolve() != root_resolved
        and root_resolved not in destination.parent.resolve().parents
    ):
        raise StorageError("destination path escapes raw storage")
    return destination


def copy_immutable(source: Path, root: Path, relative_path: str) -> FileDigest:
    """Copy exact bytes through a partial file and atomically install them once."""

    if not source.is_file() or source.is_symlink():
        raise StorageError("source must be a regular non-symlink file")
    destination = _relative_destination(root, relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_symlink():
        raise StorageError("raw destination cannot be a symlink")
    digest = hashlib.sha256()
    size = 0
    partial = destination.with_name(f"{destination.name}.partial")
    try:
        with source.open("rb") as source_handle, partial.open("wb") as partial_handle:
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                partial_handle.write(chunk)
            partial_handle.flush()
            os.fsync(partial_handle.fileno())
        if size == 0:
            raise StorageError("raw files must be non-empty")
        if destination.exists():
            existing = digest_file(destination)
            if existing.sha256 != digest.hexdigest() or existing.size != size:
                raise StorageError("raw files are immutable and differ from the requested bytes")
            partial.unlink()
        else:
            os.replace(partial, destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return FileDigest(
        path=PurePosixPath(relative_path).as_posix(), sha256=digest.hexdigest(), size=size
    )


def digest_file(path: Path) -> FileDigest:
    """Hash one non-empty regular file."""

    if not path.is_file() or path.is_symlink():
        raise StorageError("file must be a regular non-symlink file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if size == 0:
        raise StorageError("files must be non-empty")
    return FileDigest(path=path.as_posix(), sha256=digest.hexdigest(), size=size)


def clean_partials(root: Path) -> int:
    """Remove only `.partial` files below a specific raw root."""

    removed = 0
    if root.exists():
        for partial in root.rglob("*.partial"):
            if partial.is_file() and not partial.is_symlink():
                partial.unlink()
                removed += 1
    return removed
