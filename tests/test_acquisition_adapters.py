from __future__ import annotations

from pathlib import Path

import pytest

from kawaneen.acquisition.adapters import (
    AdapterError,
    HuggingFaceAdapter,
    LocalFileAdapter,
    MendeleyAdapter,
    MendeleyAdapterUnavailable,
)
from kawaneen.acquisition.models import FileExpectation, SourceSpecification


def _local_spec() -> SourceSpecification:
    return SourceSpecification(
        schema_version=1,
        source_id="arabiccr",
        version="3",
        revision="3",
        provider="mendeley_data",
        identifier="10.17632/np538c95yy.3",
        licence="CC BY 4.0",
        expected_records=1,
        files=(FileExpectation(path="ArabiCCR-dataset.csv", format="csv", expected_records=1),),
    )


def test_local_adapter_preserves_exact_filename_and_bytes(tmp_path: Path) -> None:
    source = tmp_path / "ArabiCCR-dataset.csv"
    source.write_bytes(b"id,text\n1,fiction\n")
    result = LocalFileAdapter().import_file(_local_spec(), source, tmp_path / "raw")
    assert result[0].path == "ArabiCCR-dataset.csv"
    assert (tmp_path / "raw/arabiccr/3/ArabiCCR-dataset.csv").read_bytes() == source.read_bytes()


def test_local_adapter_rejects_wrong_filename(tmp_path: Path) -> None:
    source = tmp_path / "wrong.csv"
    source.write_text("id\n1\n", encoding="utf-8")
    with pytest.raises(AdapterError, match="exactly"):
        LocalFileAdapter().import_file(_local_spec(), source, tmp_path / "raw")


def test_mendeley_adapter_requires_safe_manual_import() -> None:
    with pytest.raises(MendeleyAdapterUnavailable, match="import-local"):
        MendeleyAdapter().acquire(_local_spec(), Path("data/raw"))


def test_huggingface_adapter_uses_pinned_revision(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "cached.parquet"
    source.write_bytes(b"fiction")
    spec = SourceSpecification(
        schema_version=1,
        source_id="alarb",
        version="revision",
        revision="pinned-revision",
        provider="huggingface",
        identifier="THIQAH-RD/ALARB",
        licence="Apache-2.0",
        expected_records=1,
        files=(FileExpectation(path="data/file.parquet", format="parquet"),),
    )
    calls: list[dict[str, str]] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr("kawaneen.acquisition.adapters.hf_hub_download", fake_download)
    result = HuggingFaceAdapter().acquire(spec, tmp_path / "raw")
    assert result[0].path == "data/file.parquet"
    assert calls == [
        {
            "repo_id": "THIQAH-RD/ALARB",
            "filename": "data/file.parquet",
            "revision": "pinned-revision",
            "repo_type": "dataset",
        }
    ]


def test_huggingface_adapter_reports_download_failure(monkeypatch, tmp_path: Path) -> None:
    spec = SourceSpecification(
        schema_version=1,
        source_id="alarb",
        version="revision",
        revision="pinned-revision",
        provider="huggingface",
        identifier="THIQAH-RD/ALARB",
        licence="Apache-2.0",
        expected_records=1,
        files=(FileExpectation(path="README.md", format="text"),),
    )

    def fail_download(**_kwargs):
        raise OSError("offline")

    monkeypatch.setattr("kawaneen.acquisition.adapters.hf_hub_download", fail_download)
    with pytest.raises(AdapterError, match="download failed"):
        HuggingFaceAdapter().acquire(spec, tmp_path / "raw")
