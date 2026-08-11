import json
from pathlib import Path

from kawaneen.corpus.models import SourceFragment, SourceProvenance, UnitType
from kawaneen.corpus.orchestrator import build_statutory_review_handoff, gaps


def test_gap_report_has_15_to_25_targeted_instruments() -> None:
    rows = gaps()
    assert 15 <= len(rows) <= 25
    assert any("Companies" in row["instrument"] for row in rows)


def test_statutory_handoff_reports_exact_targets_and_preserves_only_same_ordinal(
    tmp_path: Path, monkeypatch
) -> None:
    fragments = [
        SourceFragment(
            fragment_id="one",
            provenance=SourceProvenance(
                source_id="seed",
                source_version="v1",
                source_path="rows.parquet",
                source_row=1,
                source_field="text",
            ),
            raw_label="المادة السابعة",
            law_name="Law",
            law_type="law",
            unit_type=UnitType.ARTICLE_FRAGMENT,
            text="seven",
        ),
        SourceFragment(
            fragment_id="seventeen",
            provenance=SourceProvenance(
                source_id="seed",
                source_version="v1",
                source_path="rows.parquet",
                source_row=2,
                source_field="text",
            ),
            raw_label="المادة السابعة عشرة",
            law_name="Law",
            law_type="law",
            unit_type=UnitType.ARTICLE_FRAGMENT,
            text="seventeen",
        ),
    ]
    adjudication = tmp_path / "external.jsonl"
    adjudication.write_text(
        json.dumps(
            {
                "law": {"dataset_law_name": "Law"},
                "sample_review": {
                    "results": [{"sample_role": "early", "intended_article_ordinal": 7}]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kawaneen.corpus.orchestrator._statutory_fragments", lambda: (fragments, ["Law", "Law"])
    )

    result = build_statutory_review_handoff(tmp_path / "handoff.jsonl", adjudication)
    exported = json.loads((tmp_path / "handoff.jsonl").read_text(encoding="utf-8"))

    assert result["sample_count"] == 1
    assert exported[0]["target_present"] is True
    assert [member["parsed_article_ordinal"] for member in exported[0]["members"]] == [7]
