from pathlib import Path

from kawaneen.extraction.annotation import prepare_annotation_pack
from kawaneen.extraction.readiness import build_readiness_report


def test_readiness_report_is_text_free_and_reports_zero_model_calls(tmp_path: Path) -> None:
    pack = prepare_annotation_pack(
        private_root=tmp_path / "private",
        manifest_path=tmp_path / "selection.json",
    )
    report = build_readiness_report(pack)
    assert report["selection_counts"] == {"total": 120, "dev": 80, "holdout": 40, "smoke": 10}
    assert report["deterministic_extractor_smoke_success"] is True
    assert report["hybrid_model_calls"] == 0
    assert report["holdout_sealed"] is True
    assert report["tracked_source_text_leakage"] == 0
    assert report["annotation_status_counts"]["unreviewed"] == 120
