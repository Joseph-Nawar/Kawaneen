import json
from dataclasses import replace
from pathlib import Path

import pytest

from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.extraction.annotation import (
    MAX_CANONICAL_TEXT_LENGTH,
    MIN_CANONICAL_TEXT_LENGTH,
    Phase11StructuralMetadata,
    phase11_unit_eligible,
    prepare_annotation_pack,
)
from kawaneen.extraction.candidates import CANDIDATE_REGISTRY_VERSION, build_candidate_registry
from kawaneen.extraction.orchestration import export_dev_annotation_batch_v2


def _unit(text: str) -> CanonicalUnit:
    return CanonicalUnit(
        unit_id="unit-1",
        document_id="doc-1",
        unit_type=UnitType.ARTICLE,
        text=text,
        provenance=SourceProvenance(
            source_id="saudi-moj-derived",
            source_version="version",
            source_path="data/train.parquet",
            source_row=1,
            source_field="text",
        ),
    )


def test_phase11_eligibility_requires_atomic_article_and_length_bounds() -> None:
    metadata = Phase11StructuralMetadata(
        structural_role="article",
        article_ordinal=1,
        part_index=None,
        parse_confidence="high",
    )
    assert phase11_unit_eligible(_unit("a" * MIN_CANONICAL_TEXT_LENGTH), metadata)
    assert phase11_unit_eligible(_unit("a" * MAX_CANONICAL_TEXT_LENGTH), metadata)
    assert not phase11_unit_eligible(_unit("a" * (MIN_CANONICAL_TEXT_LENGTH - 1)), metadata)
    assert not phase11_unit_eligible(_unit("a" * (MAX_CANONICAL_TEXT_LENGTH + 1)), metadata)
    assert not phase11_unit_eligible(
        _unit("a" * 100), replace(metadata, structural_role="article_part")
    )
    assert not phase11_unit_eligible(_unit("a" * 100), replace(metadata, article_ordinal=None))


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("يلتزم خلال 30 يوماً.", "30 days"),
        ("يلتزم خلال ٣٠ يوماً.", "30 days"),
        ("يلتزم خلال ثلاثين يوماً.", "30 days"),
        ("يلتزم خلال عشرين يوماً.", "20 days"),
        ("يلتزم خلال خمسة أيام.", "5 days"),
        ("يلتزم خلال سبعة أيام.", "7 days"),
        ("يلتزم خلال ستين يوماً.", "60 days"),
        ("يلتزم خلال سنة.", "1 year"),
        ("يلتزم خلال سنتين.", "2 year"),
        ("يلتزم خلال شهر.", "1 month"),
        ("يلتزم خلال شهرين.", "2 month"),
    ],
)
def test_arabic_duration_candidates_are_exact_and_conservatively_normalized(
    text: str, value: str
) -> None:
    registry = build_candidate_registry(text, canonical_unit_id="u", document_id="d")
    candidates = [item for item in registry.candidates if item.candidate_type.value == "temporal"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert text[candidate.span.start_char : candidate.span.end_char] == candidate.raw_exact_text
    assert candidate.normalized.normalized_value == value


def test_regulation_candidates_are_bounded_and_do_not_follow_bare_lexical_tokens() -> None:
    text = (
        "نظام التنفيذ. نظام المرافعات الشرعية. نظام الجلسة تبدأ. "
        "نظام دون تقديم. لائحة دعوى أو جواب."
    )
    registry = build_candidate_registry(text, canonical_unit_id="u", document_id="d")
    values = [
        item.raw_exact_text
        for item in registry.candidates
        if item.candidate_type.value == "regulation"
    ]
    assert values == ["نظام التنفيذ", "نظام المرافعات الشرعية"]
    for candidate in registry.candidates:
        assert text[candidate.span.start_char : candidate.span.end_char] == candidate.raw_exact_text


def test_v2_selection_and_batch_are_fresh_and_unreviewed(tmp_path: Path) -> None:
    pack = prepare_annotation_pack(
        private_root=tmp_path / "annotations",
        manifest_path=tmp_path / "selection.json",
    )
    assert pack["selection_version"] == "phase11-selection-v2"
    assert pack["selection_fingerprint"] != "phase11-selection-v1"
    records = pack["records"]
    rows = json.loads((tmp_path / "selection.json").read_text())["rows"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    assert len(dev_rows) == 80
    assert len(holdout_rows) == 40
    assert {row["document_id"] for row in dev_rows}.isdisjoint(
        row["document_id"] for row in holdout_rows
    )
    assert max(len(record.canonical_text) for record in records) <= MAX_CANONICAL_TEXT_LENGTH
    assert any("low_signal" in row["strata"] for row in rows)
    assert all(record.annotation_status == "unreviewed" for record in records)
    assert all(record.annotation_provenance == "unreviewed" for record in records)
    assert all(record.human_annotations is None for record in records)
    assert all(record.human_verified is False for record in records)
    result = export_dev_annotation_batch_v2(
        annotation_root=tmp_path / "annotations",
        selection_manifest_path=tmp_path / "selection.json",
        output_path=tmp_path / "batch-v2.json",
    )
    assert result["dev_records"] == 80
    assert result["holdout_records"] == 0


def test_candidate_registry_version_is_bumped() -> None:
    assert CANDIDATE_REGISTRY_VERSION == "phase11-candidates-v3"


def test_money_percentage_and_offsets_regression() -> None:
    text = "الحد 1250 SAR ونسبة 15% دون تغيير النص."
    registry = build_candidate_registry(text, canonical_unit_id="u", document_id="d")
    assert [item.raw_exact_text for item in registry.candidates] == ["1250 SAR", "15%"]
    assert [text[item.span.start_char : item.span.end_char] for item in registry.candidates] == [
        item.raw_exact_text for item in registry.candidates
    ]
