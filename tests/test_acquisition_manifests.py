from __future__ import annotations

from pathlib import Path

from kawaneen.acquisition.manifests import build_manifests, validate_manifests
from kawaneen.acquisition.models import FileDigest, IntegrityResult, PrivacyResult
from kawaneen.acquisition.privacy import summarize_privacy


def test_manifests_are_deterministic_and_validate(tmp_path: Path) -> None:
    result = IntegrityResult(
        source_id="alarb",
        files=(FileDigest(path="data/raw/alarb/test/file.parquet", size=4, sha256="a" * 64),),
        row_counts={"file.parquet": 1},
        schema_fingerprints={"file.parquet": "b" * 64},
        physical_duplicate_count=0,
        duplicate_row_count=0,
        split_overlap_count=0,
    )
    privacy = PrivacyResult(source_id="alarb", finding_count=0)
    build_manifests(
        result,
        privacy,
        summarize_privacy(privacy),
        "test",
        "evaluation",
        tmp_path,
    )
    validate_manifests(tmp_path)
    assert '"schema_version": 2' in (tmp_path / "acquisition_lock.json").read_text(encoding="utf-8")
    assert '"canonical_source": "unspecified"' in (tmp_path / "acquisition_lock.json").read_text(
        encoding="utf-8"
    )
    assert (
        (tmp_path / "raw_file_manifest.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .startswith("schema_version,source_id,version,path,size,sha256")
    )
    assert (
        (tmp_path / "privacy_review_status.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .startswith("schema_version,source_id,status,finding_count,affected_record_count")
    )
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    build_manifests(
        result,
        privacy,
        summarize_privacy(privacy),
        "test",
        "evaluation",
        tmp_path,
    )
    second = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert first == second
    for path in tmp_path.iterdir():
        assert str(path).startswith(str(tmp_path))
        assert str(tmp_path).encode() not in path.read_bytes()
        assert b"fiction" not in path.read_bytes()
