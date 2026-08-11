import json

import pytest

from kawaneen.corpus.models import SourceFragment, SourceProvenance, UnitType
from kawaneen.corpus.statutory import (
    build_statutory_review_samples,
    classify_all,
    classify_fragment_group,
    duplicate_diagnostics,
    parse_article_label,
)


def fragment(row: int, label: str, text: str, part: int | None = None) -> SourceFragment:
    return SourceFragment(
        fragment_id=f"f{row}",
        provenance=SourceProvenance(
            source_id="fixture",
            source_version="1",
            source_path="x.parquet",
            source_row=row,
            source_field="text",
        ),
        raw_label=label,
        derived_article_ordinal=parse_article_label(label).ordinal,
        explicit_part=part,
        unit_type=UnitType.ARTICLE_FRAGMENT,
        text=text,
    )


def test_article_label_parser_preserves_raw_label_and_extracts_structural_numbers() -> None:
    assert parse_article_label("المادة ١٢").ordinal == 12
    assert parse_article_label("المادة الثانية").ordinal == 2
    assert parse_article_label("Article 12 part 2").part == 2


@pytest.mark.parametrize(
    ("label", "ordinal"),
    [
        (f"المادة {word}", number)
        for word, number in (
            ("الرابعة", 4),
            ("الرابعة عشرة", 14),
            ("الرابعة والعشرون", 24),
            ("الرابعة والثلاثون", 34),
            ("الرابعة والأربعون", 44),
            ("الرابعة والخمسون", 54),
            ("الرابعة والستون", 64),
            ("الرابعة والسبعون", 74),
            ("الرابعة والثمانون", 84),
            ("الرابعة والتسعون", 94),
            ("السابعة", 7),
            ("السابعة عشرة", 17),
            ("السابعة والعشرون", 27),
            ("السابعة والثلاثون", 37),
            ("السابعة والأربعون", 47),
            ("السابعة والخمسون", 57),
            ("السابعة والستون", 67),
            ("السابعة والسبعون", 77),
            ("السابعة والثمانون", 87),
            ("العشرون", 20),
            ("الأربعون", 40),
            ("المائة", 100),
        )
    ],
)
def test_article_label_parser_distinguishes_full_arabic_ordinals(label: str, ordinal: int) -> None:
    parsed = parse_article_label(label)
    assert parsed.ordinal == ordinal
    assert parsed.article_parse_confidence.value == "high"
    assert parsed.article_label_structural_key


@pytest.mark.parametrize(
    ("label", "ordinal"),
    [
        ("المادة 1", 1),
        ("المادة \u0661\u0660\u0661", 101),
        ("المادة 201", 201),
        ("المادة \u0663\u0660\u0661", 301),
        ("المادة 701", 701),
        ("المادة 99", 99),
        ("المادة 109", 109),
        ("المادة 119", 119),
        ("المادة 149", 149),
        ("المادة 169", 169),
        ("المادة 209", 209),
    ],
)
def test_article_label_parser_accepts_western_and_arabic_indic_digits(
    label: str, ordinal: int
) -> None:
    assert parse_article_label(label).ordinal == ordinal


@pytest.mark.parametrize(
    ("label", "ordinal"),
    [
        ("المادة الحاديةعشرة", 11),
        ("المادة الرابعةعشرة", 14),
        ("المادة الرابعةوالعشرون", 24),
        ("المادة الأولى بعد المائة", 101),
        ("المادة الأولى بعدالمائة", 101),
        ("المادة الثانية بعد المائتين", 202),
        ("المادة الثانية بعدالمائتين", 202),
        ("المادة الثالثة بعد الثلاثمائة", 303),
        ("المادة الثالثة بعدالثلاثمائة", 303),
    ],
)
def test_article_label_parser_accepts_joined_and_hundred_forms(label: str, ordinal: int) -> None:
    assert parse_article_label(label).ordinal == ordinal


def test_part_marker_never_replaces_article_ordinal() -> None:
    second = parse_article_label("المادة الستون (جزء 2)")
    third = parse_article_label("المادة الستون (جزء 3)")
    assert second.ordinal == third.ordinal == 60
    assert second.part == 2
    assert third.part == 3
    assert second.article_label_structural_key == third.article_label_structural_key


@pytest.mark.parametrize("label", ["فقرة 1", "المادة غير معروفة", "", "المادة (جزء 2)"])
def test_malformed_or_unknown_labels_fail_closed(label: str) -> None:
    parsed = parse_article_label(label)
    assert parsed.ordinal is None
    assert parsed.article_parse_confidence.value == "unresolved"
    assert parsed.article_label_structural_key is None


def test_status_suffix_is_structural_metadata_not_source_text() -> None:
    parsed = parse_article_label("المادة الحادية عشرة معدلة")
    assert parsed.ordinal == 11
    assert parsed.article_status_marker == "معدلة"
    assert parsed.raw_label == "المادة الحادية عشرة معدلة"


def test_distinct_full_ordinals_never_share_structural_keys() -> None:
    keys = {
        parse_article_label(label).article_label_structural_key
        for label in ("المادة الرابعة", "المادة الرابعة عشرة", "المادة الرابعة والعشرون")
    }
    assert len(keys) == 3


def test_classify_all_does_not_merge_distinct_full_ordinals() -> None:
    first = fragment(1, "المادة الرابعة", "A")
    second = fragment(2, "المادة الرابعة عشرة", "B")
    groups = classify_all((first, second), ("Law", "Law"))
    assert len(groups) == 2
    assert {group.raw_article_label for group in groups} == {
        "المادة الرابعة",
        "المادة الرابعة عشرة",
    }


def test_duplicate_group_merges_only_explicit_parts() -> None:
    explicit = classify_fragment_group(
        "Law",
        "Article 1",
        [fragment(1, "Article 1 part 1", "A", 1), fragment(2, "Article 1 part 2", "B", 2)],
    )
    ambiguous = classify_fragment_group(
        "Law", "Article 1", [fragment(1, "Article 1", "A"), fragment(2, "Article 1", "B")]
    )
    assert explicit.status.value == "explicit_fragment_series"
    assert ambiguous.status.value in {"conflicting_duplicate", "unresolved"}
    assert len(ambiguous.fragment_ids) == 2


def test_duplicate_diagnostics_are_sanitized_and_conservative() -> None:
    items = [fragment(1, "Article 1", "A"), fragment(3, "Article 1", "B")]
    group = classify_fragment_group("Law", "Article 1", items)
    result = duplicate_diagnostics(items, (group,))
    assert result["duplicate_group_count"] == 1
    assert result["genuine_conflict_candidates"] == 1
    assert result["ambiguous_continuation_candidates"] == 0
    assert "text" not in json.dumps(result)


def test_review_target_contains_only_its_full_article_ordinal() -> None:
    """A partial-label grouping bug must not leak 17/27 into Article 7."""

    samples = build_statutory_review_samples(
        "Law",
        (
            fragment(1, "المادة السابعة", "seven"),
            fragment(2, "المادة السابعة عشرة", "seventeen"),
            fragment(3, "المادة السابعة والعشرون", "twenty-seven"),
            fragment(4, "المادة السابعة (جزء 2)", "seven part two", 2),
            fragment(5, "المادة السابعة معدلة", "seven amended"),
        ),
        {"early": 7},
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample["requested_article_ordinal"] == 7
    assert {member["parsed_article_ordinal"] for member in sample["members"]} == {7}
    assert sample["target_present"] is True


def test_review_target_changes_metadata_when_requested_ordinal_is_unavailable() -> None:
    samples = build_statutory_review_samples(
        "Law",
        (fragment(1, "المادة الثالثة", "three"),),
        {"early": 7},
    )

    sample = samples[0]
    assert sample["requested_article_ordinal"] == 3
    assert sample["selection_resolution"] == "fallback_exact_available_ordinal"
    assert sample["target_present"] is True
