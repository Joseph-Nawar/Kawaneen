from pathlib import Path

from kawaneen.extraction.annotation import (
    ANNOTATION_ROOT,
    SELECTION_MANIFEST_PATH,
    prepare_annotation_pack,
    validate_annotation_record,
)


def test_annotation_selection_is_reproducible_and_protected(tmp_path: Path) -> None:
    first = prepare_annotation_pack(
        private_root=tmp_path / "private-1",
        manifest_path=tmp_path / "manifest-1.json",
    )
    second = prepare_annotation_pack(
        private_root=tmp_path / "private-2",
        manifest_path=tmp_path / "manifest-2.json",
    )
    assert first["selection_counts"] == {"total": 120, "dev": 80, "holdout": 40, "smoke": 10}
    assert first["selection_fingerprint"] == second["selection_fingerprint"]
    assert first["document_disjoint"] is True
    assert first["selection_counts"]["smoke"] == 10
    assert SELECTION_MANIFEST_PATH.as_posix().endswith("phase11_annotation_selection.json")
    assert ANNOTATION_ROOT.as_posix().endswith("phase11_extraction/annotations")


def test_new_annotation_records_are_not_human_gold(tmp_path: Path) -> None:
    result = prepare_annotation_pack(
        private_root=tmp_path / "private",
        manifest_path=tmp_path / "manifest.json",
    )
    record = result["records"][0]
    assert record.annotation_status == "unreviewed"
    assert record.human_verified is False
    assert validate_annotation_record(record, {record.canonical_unit_id}) == []
