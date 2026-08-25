from kawaneen.api.contracts import DocumentUnit
from kawaneen.extraction.contracts import (
    Candidate,
    CandidateType,
    ExactSourceSpan,
    NormalizationStatus,
    NormalizedRepresentation,
)
from kawaneen.ui.demo import DemoClient
from kawaneen.ui.formatting import highlight_literal, locate_quote, text_direction
from kawaneen.ui.presentation import (
    document_page_bounds,
    extract_presentation_rows,
    filter_returned_evidence,
    inspect_verified_quote,
)


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


def test_returned_document_filter_preserves_original_api_ranking() -> None:
    results = (
        DemoClient().search("appeal deadline").results
        + DemoClient().search("ما هي مدة الاعتراض؟").results
    )

    filtered = filter_returned_evidence(results, {"Employment Procedures Regulation"})

    assert [item.rank for item in filtered] == [1]
    assert filtered[0].document_title == "Employment Procedures Regulation"


def test_verified_quote_highlights_only_an_exact_canonical_match_safely() -> None:
    unit = DocumentUnit(
        unit_id="u-1",
        ordinal=1,
        unit_type="article",
        text='يجب السداد خلال ثلاثين يوماً. <script>alert("x")</script>',
        heading_path=("المادة 1",),
    )

    rendered = inspect_verified_quote(unit, "خلال ثلاثين يوماً")

    assert 'class="verified-quote"' in rendered
    assert "خلال ثلاثين يوماً" in rendered
    assert "<script>" not in rendered
    assert "canonical unit: u-1" in rendered


def test_document_page_bounds_expose_visible_range_and_navigation() -> None:
    assert document_page_bounds(offset=0, limit=2, total=5) == (1, 2, 5, False, True)
    assert document_page_bounds(offset=2, limit=2, total=5) == (3, 4, 5, True, True)
    assert document_page_bounds(offset=4, limit=2, total=5) == (5, 5, 5, True, False)


def test_extraction_presentation_rows_include_structure_source_and_segment() -> None:
    response = DemoClient().extract("The party must pay within thirty days.")
    deadline = Candidate(
        candidate_id="T001",
        candidate_type=CandidateType.TEMPORAL,
        span=ExactSourceSpan(
            text="thirty days",
            start_char=26,
            end_char=38,
            canonical_unit_id="api-request",
            document_id="api-request",
        ),
        raw_exact_text="thirty days",
        normalized=NormalizedRepresentation(normalized_value="30 days"),
        normalization_status=NormalizationStatus.NORMALIZED,
    )
    updated_result = response.result.model_copy(update={"deadlines": (deadline,)})
    response = response.model_copy(update={"result": updated_result})

    rows = extract_presentation_rows("segment-001", response)

    fields = {row["field"] for row in rows}
    assert {"summary", "obligation", "deadline", "source"} <= fields
    assert all(row["segment_id"] == "segment-001" for row in rows)
