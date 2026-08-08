from __future__ import annotations

from pathlib import Path

import pytest

import kawaneen.acquisition.orchestrator as orchestrator
from kawaneen.acquisition.models import (
    AcquisitionPurpose,
    FileDigest,
    FileExpectation,
    IntegrityResult,
    PrivacyResult,
    SourceSpecification,
)


def _spec() -> SourceSpecification:
    return SourceSpecification(
        schema_version=1,
        source_id="arabiccr",
        version="3",
        revision="3",
        provider="mendeley_data",
        identifier="10.17632/np538c95yy.3",
        licence="CC BY 4.0",
        expected_records=1,
        files=(FileExpectation(path="ArabiCCR-dataset.csv", format="csv"),),
    )


def _integrity() -> IntegrityResult:
    return IntegrityResult(
        source_id="arabiccr",
        files=(FileDigest(path="ArabiCCR-dataset.csv", size=1, sha256="a" * 64),),
        physical_duplicate_count=0,
        duplicate_row_count=0,
        split_overlap_count=0,
    )


def test_plan_and_status_are_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(orchestrator, "load_specifications", lambda: {"arabiccr": _spec()})
    monkeypatch.setattr(orchestrator, "DEFAULT_RAW_ROOT", tmp_path / "raw")
    assert orchestrator.plan()[0]["source_id"] == "arabiccr"
    assert orchestrator.status()[0]["raw_present"] is False


def test_mendeley_acquisition_requires_manual_import(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(orchestrator, "_spec", lambda _source: _spec())
    with pytest.raises(orchestrator.AdapterError, match="import-local"):
        orchestrator.acquire_source("arabiccr", AcquisitionPurpose.LOCAL_RESEARCH, tmp_path)


def test_orchestration_builds_and_rebuilds_sanitized_manifests(monkeypatch, tmp_path: Path) -> None:
    spec = _spec()
    monkeypatch.setattr(orchestrator, "_spec", lambda _source: spec)
    monkeypatch.setattr(orchestrator, "DEFAULT_RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(orchestrator, "DEFAULT_MANIFEST_ROOT", tmp_path / "manifests")
    monkeypatch.setattr(orchestrator, "verify_specification", lambda *_args: _integrity())
    monkeypatch.setattr(
        orchestrator,
        "audit_source",
        lambda *_args: PrivacyResult(source_id="arabiccr", finding_count=0),
    )
    calls: list[str] = []
    monkeypatch.setattr(orchestrator, "build_manifests", lambda *_args: calls.append("built"))
    orchestrator.build_manifest("arabiccr")
    assert calls == ["built"]
    monkeypatch.setattr(orchestrator, "load_specifications", lambda: {"arabiccr": spec})
    (tmp_path / "raw" / "arabiccr" / "3").mkdir(parents=True)
    assert orchestrator.rebuild_auto() == [{"source_id": "arabiccr", "rebuilt": True}]


def test_manifest_and_audit_wrappers_delegate(monkeypatch, tmp_path: Path) -> None:
    spec = _spec()
    monkeypatch.setattr(orchestrator, "_spec", lambda _source: spec)
    monkeypatch.setattr(orchestrator, "DEFAULT_RAW_ROOT", tmp_path / "raw")
    result = PrivacyResult(source_id="arabiccr", finding_count=0)
    monkeypatch.setattr(orchestrator, "screen_privacy", lambda *_args: result)
    written: list[Path] = []
    monkeypatch.setattr(
        orchestrator, "write_private_review_bundle", lambda _result, path: written.append(path)
    )
    assert orchestrator.audit_source("arabiccr") == result
    assert written == [orchestrator.DEFAULT_PRIVATE_ROOT]
    monkeypatch.setattr(orchestrator, "validate_manifests", lambda path: written.append(path))
    orchestrator.validate_manifest()
    assert written[-1] == orchestrator.DEFAULT_MANIFEST_ROOT
