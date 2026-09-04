from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_hf_bootstrap_module() -> ModuleType:
    path = REPO_ROOT / "docker" / "hf-model-init.py"
    spec = importlib.util.spec_from_file_location("kawaneen_hf_model_init", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_snapshot(path: Path, *, metadata: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if metadata:
        (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def test_bootstrap_passes_exact_frozen_dense_and_reranker_identities(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> Path:
        calls.append(kwargs)
        model_id = str(kwargs["repo_id"])
        return _write_snapshot(tmp_path / "cache" / model_id.replace("/", "--"))

    result = load_hf_bootstrap_module().bootstrap_models(REPO_ROOT / "data", tmp_path / "cache", download)

    assert [(call["repo_id"], call["revision"]) for call in calls] == [
        ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
        ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    ]
    assert [snapshot.model_id for snapshot in result] == [
        "BAAI/bge-m3",
        "BAAI/bge-reranker-v2-m3",
    ]
    assert all(snapshot.path.is_dir() for snapshot in result)


def test_bootstrap_is_safe_to_repeat_without_forced_download(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> Path:
        calls.append(kwargs)
        model_id = str(kwargs["repo_id"])
        return _write_snapshot(tmp_path / "cache" / model_id.replace("/", "--"))

    module = load_hf_bootstrap_module()
    module.bootstrap_models(REPO_ROOT / "data", tmp_path / "cache", download)
    module.bootstrap_models(REPO_ROOT / "data", tmp_path / "cache", download)

    assert len(calls) == 4
    assert all(call["local_files_only"] is False for call in calls)
    assert all("force_download" not in call for call in calls)


def test_bootstrap_propagates_download_failure(tmp_path: Path) -> None:
    def download(**_: object) -> Path:
        raise RuntimeError("frozen revision unavailable")

    with pytest.raises(RuntimeError, match="frozen revision unavailable"):
        load_hf_bootstrap_module().bootstrap_models(REPO_ROOT / "data", tmp_path / "cache", download)


def test_bootstrap_rejects_snapshot_without_model_metadata(tmp_path: Path) -> None:
    def download(**kwargs: object) -> Path:
        return _write_snapshot(tmp_path / "cache" / str(kwargs["repo_id"]).replace("/", "--"), metadata=False)

    with pytest.raises(RuntimeError, match="metadata"):
        load_hf_bootstrap_module().bootstrap_models(REPO_ROOT / "data", tmp_path / "cache", download)
