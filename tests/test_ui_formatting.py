from kawaneen.api.contracts import DocumentUnit
from kawaneen.ui.formatting import highlight_literal, locate_quote, text_direction


def test_highlight_literal_escapes_markup_and_marks_literal_matches() -> None:
    rendered = highlight_literal('<script>alert("x")</script> Appeal', "appeal")

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert '<mark class="query-hit">Appeal</mark>' in rendered


def test_text_direction_handles_arabic_and_mixed_english() -> None:
    assert text_direction("ما هي مدة الاعتراض؟") == "rtl"
    assert text_direction("appeal deadline") == "ltr"
    assert text_direction("Article 12 — مدة الاعتراض") == "rtl"


def test_locate_quote_returns_canonical_unit_and_offsets() -> None:
    units = (
        DocumentUnit(
            unit_id="u-1",
            ordinal=1,
            unit_type="article",
            text="يلتزم الطرف بالسداد خلال ثلاثين يوماً.",
            heading_path=(),
        ),
    )

    location = locate_quote(units, "خلال ثلاثين يوماً")

    assert location is not None
    assert location.unit_id == "u-1"
    assert units[0].text[location.start_char : location.end_char] == "خلال ثلاثين يوماً"
