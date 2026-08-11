import hashlib
import json
import sys
from pathlib import Path

import pytest

import kawaneen.parsing.anchored_gold as anchored_gold
from kawaneen.parsing.anchored_gold import (
    AmbiguousAnchorError,
    GoldIntegrityError,
    find_unique_text_anchor,
    pdf_bottom_left_to_top_left,
    scale_box,
    union_boxes,
    validate_anchored_gold,
)


def test_pdf_bottom_left_box_converts_to_top_left() -> None:
    assert pdf_bottom_left_to_top_left((10, 20, 110, 40), page_height=800) == (
        10.0,
        760.0,
        110.0,
        780.0,
    )


def test_box_scaling_preserves_relative_position() -> None:
    assert scale_box((10, 20, 110, 40), 600, 800, 1200, 1600) == (
        20.0,
        40.0,
        220.0,
        80.0,
    )


def test_multiline_span_union_contains_each_line_box() -> None:
    assert union_boxes(((10, 20, 100, 30), (20, 35, 120, 45))) == (
        10.0,
        20.0,
        120.0,
        45.0,
    )


def test_text_anchor_fails_closed_on_ambiguous_matches() -> None:
    with pytest.raises(AmbiguousAnchorError):
        find_unique_text_anchor(
            "المادة الأولى\nالمادة الأولى",  # noqa: RUF001
            "المادة الأولى",
            page_width=600,
            page_height=800,
        )


def test_text_anchor_unions_character_geometry_and_locator_whitespace() -> None:
    boxes = [(index, 10, index + 1, 20) for index in range(5)]
    assert find_unique_text_anchor(
        "أ ب ج",
        "أبج",
        page_width=100,
        page_height=100,
        character_boxes=boxes,
    ) == (0.0, 10.0, 5.0, 20.0)


def test_geometry_helpers_reject_invalid_dimensions() -> None:
    with pytest.raises(GoldIntegrityError):
        pdf_bottom_left_to_top_left((0, 0, 1, 1), page_height=0)
    with pytest.raises(GoldIntegrityError):
        scale_box((0, 0, 1, 1), 0, 1, 1, 1)
    with pytest.raises(GoldIntegrityError):
        union_boxes(())


def test_internal_span_locator_handles_occurrence_and_fuzzy_source_text() -> None:
    assert anchored_gold._reference_span("abc abc", "abc", occurrence_index=1) == (3, 6, True)
    start, end, ambiguous = anchored_gold._reference_span("abcdefgh12345XXXX", "abcdefgh12345YYYY")
    assert (start, end, ambiguous) == (0, 17, False)
    with pytest.raises(AmbiguousAnchorError):
        anchored_gold._reference_span("abc abc", "abc")
    with pytest.raises(GoldIntegrityError, match="similar"):
        anchored_gold._reference_span("abcdefghZZZZZZ", "abcdefgh123456")


def test_internal_index_mapping_handles_equal_and_non_equal_opcodes() -> None:
    opcodes = [("equal", 0, 2, 0, 2), ("replace", 2, 4, 2, 3)]
    assert anchored_gold._map_index(1, opcodes) == 1
    assert anchored_gold._map_index(3, opcodes) == 2
    assert anchored_gold._map_index(8, opcodes) == 3
    assert anchored_gold._map_index(8, ()) == 8


def test_pdfium_anchor_uses_source_locator_and_converts_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTextPage:
        def count_chars(self) -> int:
            return 3

        def get_text_range(self, index: int | None = None, count: int | None = None) -> str:
            text = "abc"
            return text if index is None else text[index : index + (count or 1)]

        def get_charbox(self, index: int) -> tuple[float, float, float, float]:
            return (float(index), 10.0, float(index + 1), 20.0)

    class FakePage:
        def get_size(self) -> tuple[float, float]:
            return (100.0, 100.0)

        def get_textpage(self) -> FakeTextPage:
            return FakeTextPage()

    class FakeDocument:
        def __init__(self, _path: str) -> None:
            pass

        def __getitem__(self, _index: int) -> FakePage:
            return FakePage()

    monkeypatch.setitem(
        sys.modules, "pypdfium2", type("FakePdfium", (), {"PdfDocument": FakeDocument})
    )
    path = tmp_path / "source.pdf"
    path.write_bytes(b"not parsed by fake PDFium")
    assert anchored_gold._pdfium_anchor(path, 1, "ignored", source_locator_text="abc") == (
        100.0,
        100.0,
        (0.0, 80.0, 3.0, 90.0),
        False,
    )


def test_pdfium_anchor_maps_pypdf_text_and_falls_back_to_pdfium_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTextPage:
        def count_chars(self) -> int:
            return 3

        def get_text_range(self, index: int | None = None, count: int | None = None) -> str:
            text = "abc"
            return text if index is None else text[index : index + (count or 1)]

        def get_charbox(self, index: int) -> tuple[float, float, float, float]:
            return (float(index), 10.0, float(index + 1), 20.0)

    class FakePage:
        def get_size(self) -> tuple[float, float]:
            return (100.0, 100.0)

        def get_textpage(self) -> FakeTextPage:
            return FakeTextPage()

    class FakeDocument:
        def __init__(self, _path: str) -> None:
            pass

        def __getitem__(self, _index: int) -> FakePage:
            return FakePage()

    class FakePdfPage:
        def extract_text(self) -> str:
            return "abc"

    class FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePdfPage()]

    monkeypatch.setitem(
        sys.modules, "pypdfium2", type("FakePdfium", (), {"PdfDocument": FakeDocument})
    )
    monkeypatch.setitem(sys.modules, "pypdf", type("FakePypdf", (), {"PdfReader": FakeReader}))
    path = tmp_path / "source.pdf"
    path.write_bytes(b"source")
    assert anchored_gold._pdfium_anchor(path, 1, "abc")[:3] == (
        100.0,
        100.0,
        (0.0, 80.0, 3.0, 90.0),
    )

    class MismatchReader:
        def __init__(self, _path: str) -> None:
            self.pages = [type("Page", (), {"extract_text": lambda _self: "different"})()]

    monkeypatch.setitem(sys.modules, "pypdf", type("FakePypdf", (), {"PdfReader": MismatchReader}))
    assert anchored_gold._pdfium_anchor(path, 1, "abc")[:3] == (
        100.0,
        100.0,
        (0.0, 80.0, 3.0, 90.0),
    )


def test_gold_validation_rejects_page_and_hash_mismatch() -> None:
    record = {
        "region_id": "r1",
        "source_pdf_sha256": "a" * 64,
        "page_number": 1,
        "gold_text": "نص",
        "region_type": "paragraph",
        "semantic_article_number": None,
        "bounding_box": [0, 0, 10, 10],
        "page_width": 100,
        "page_height": 100,
        "coordinate_origin": "top-left",
        "coordinate_system": "canonical_top_left_points",
        "anchoring_method": "pdfium_text_geometry",
        "anchoring_confidence": "high",
        "review_provenance": {"human_verified": False},
        "page_id": "p1",
        "source_pdf_filename": "source.pdf",
    }
    with pytest.raises(GoldIntegrityError, match="page IDs"):
        validate_anchored_gold(
            [record],
            expected_page_count=1,
            expected_region_count=1,
            source_hashes={"source.pdf": "b" * 64},
            expected_pages={"p2"},
        )


def test_gold_validation_rejects_hash_mismatch_after_page_match() -> None:
    record = {
        "region_id": "r1",
        "source_pdf_sha256": "a" * 64,
        "page_number": 1,
        "gold_text": "نص",
        "region_type": "paragraph",
        "semantic_article_number": None,
        "bounding_box": [0, 0, 10, 10],
        "page_width": 100,
        "page_height": 100,
        "coordinate_origin": "top-left",
        "coordinate_system": "canonical_top_left_points",
        "anchoring_method": "pdfium_text_geometry",
        "anchoring_confidence": "high",
        "review_provenance": {"human_verified": False},
        "page_id": "p1",
        "source_pdf_filename": "source.pdf",
    }
    with pytest.raises(GoldIntegrityError, match="hash mismatch"):
        validate_anchored_gold(
            [record],
            expected_page_count=1,
            expected_region_count=1,
            source_hashes={"source.pdf": "b" * 64},
            expected_pages={"p1"},
        )


def test_gold_validation_rejects_invalid_or_out_of_bounds_box() -> None:
    record = {
        "region_id": "r1",
        "source_pdf_sha256": "a" * 64,
        "page_number": 1,
        "gold_text": "نص",
        "region_type": "paragraph",
        "semantic_article_number": None,
        "bounding_box": [0, 0, 110, 10],
        "page_width": 100,
        "page_height": 100,
        "coordinate_origin": "top-left",
        "coordinate_system": "canonical_top_left_points",
        "anchoring_method": "pdfium_text_geometry",
        "anchoring_confidence": "high",
        "review_provenance": {"human_verified": False},
        "page_id": "p1",
        "source_pdf_filename": "source.pdf",
    }
    with pytest.raises(GoldIntegrityError, match="bounds"):
        validate_anchored_gold(
            [record],
            expected_page_count=1,
            expected_region_count=1,
            source_hashes={"source.pdf": "a" * 64},
            expected_pages={"p1"},
        )


def test_gold_validation_requires_unique_region_ids_and_nonempty_text() -> None:
    base = {
        "region_id": "r1",
        "source_pdf_sha256": "a" * 64,
        "page_number": 1,
        "gold_text": "",
        "region_type": "paragraph",
        "semantic_article_number": None,
        "bounding_box": [0, 0, 10, 10],
        "page_width": 100,
        "page_height": 100,
        "coordinate_origin": "top-left",
        "coordinate_system": "canonical_top_left_points",
        "anchoring_method": "pdfium_text_geometry",
        "anchoring_confidence": "high",
        "review_provenance": {"human_verified": False},
        "page_id": "p1",
        "source_pdf_filename": "source.pdf",
    }
    with pytest.raises(GoldIntegrityError, match="non-empty"):
        validate_anchored_gold(
            [base, {**base}],
            expected_page_count=1,
            expected_region_count=2,
            source_hashes={"source.pdf": "a" * 64},
            expected_pages={"p1"},
        )


def test_private_gold_source_hash_is_sha256(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "41cf6794ba4200b839c53531555f0f3998df4cbb01a4d5cb0b94e3ca5e23947d"
    )


def test_build_anchored_gold_uses_external_text_and_independent_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = source_dir / "source.pdf"
    source.write_bytes(b"private test source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    selection_rows = []
    external_rows = []
    annotation_rows = []
    for page_number in range(1, 31):
        page_id = f"page-{page_number:02d}"
        category = "image_only_scan" if page_number == 1 else "born_digital_arabic"
        selection_rows.append(
            {
                "id": page_id,
                "source_pdf": "source.pdf",
                "source_page": page_number,
                "category": category,
            }
        )
        count = 3 if page_number > 12 else 4
        regions = []
        for region_number in range(1, count + 1):
            region_id = f"{page_id}-r{region_number}"
            text = f"gold {page_number} {region_number}"
            regions.append(
                {
                    "region_id": region_id,
                    "adjudicated": {
                        "text": text,
                        "region_type": "paragraph",
                        "expected_semantic_article_number": None,
                    },
                    "provenance": {
                        "source_pdf_sha256": digest,
                        "reviewer_type": "independent_ai_visual_review",
                    },
                }
            )
            if page_number == 1:
                annotation_rows.append(
                    {
                        "page_id": page_id,
                        "region_id": region_id,
                        "gold_text": text,
                        "bounding_box": [region_number, 1, region_number + 1, 2],
                        "page_width": 100,
                        "page_height": 100,
                        "anchoring_confidence": "reviewed_visual",
                        "annotation_tool": "private_manual_canvas_v1",
                    }
                )
        external_rows.append({"page_id": page_id, "regions": regions})
    selection_path = tmp_path / "selection.json"
    external_path = tmp_path / "external.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    output_path = tmp_path / "anchored.jsonl"
    selection_path.write_text(json.dumps({"selection": selection_rows}), encoding="utf-8")
    external_path.write_text(
        "".join(json.dumps(row) + "\n" for row in external_rows), encoding="utf-8"
    )
    annotations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in annotation_rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        anchored_gold,
        "_pdfium_anchor",
        lambda *args, **kwargs: (100.0, 100.0, (10.0, 10.0, 20.0, 20.0), False),
    )
    result = anchored_gold.build_anchored_gold(
        selection_path,
        external_path,
        source_dir,
        output_path,
        sama_annotations_path=annotations_path,
        born_digital_overrides_path=None,
    )
    assert (result.page_count, result.region_count) == (30, 102)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["gold_text"] == "gold 1 1"
    assert records[0]["anchoring_method"] == "independent_ai_visual_anchor"
    assert records[-1]["anchoring_method"] == "pdfium_text_geometry"


def test_build_anchored_gold_supports_independent_manual_box_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = source_dir / "source.pdf"
    source.write_bytes(b"source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    selection = []
    external = []
    for index in range(30):
        page_id = f"p{index}"
        selection.append(
            {
                "id": page_id,
                "source_pdf": "source.pdf",
                "source_page": 1,
                "category": "born_digital_arabic",
            }
        )
        external.append(
            {
                "page_id": page_id,
                "regions": [
                    {
                        "region_id": f"{page_id}-r",
                        "adjudicated": {
                            "text": "text",
                            "region_type": "paragraph",
                            "expected_semantic_article_number": None,
                        },
                        "provenance": {
                            "source_pdf_sha256": digest,
                            "reviewer_type": "independent_ai_visual_review",
                        },
                    }
                ]
                * (4 if index < 12 else 3),
            }
        )
    selection_path = tmp_path / "selection.json"
    external_path = tmp_path / "external.jsonl"
    overrides_path = tmp_path / "overrides.jsonl"
    output_path = tmp_path / "anchored.jsonl"
    selection_path.write_text(json.dumps({"selection": selection}), encoding="utf-8")
    external_path.write_text("".join(json.dumps(row) + "\n" for row in external), encoding="utf-8")
    # This fixture intentionally uses distinct region IDs below; duplicate IDs in
    # one page would be rejected by the validator.
    fixed_external = []
    for page in external:
        regions = []
        for number, region in enumerate(page["regions"], 1):
            item = dict(region)
            item["region_id"] = f"{page['page_id']}-r{number}"
            regions.append(item)
        fixed_external.append({"page_id": page["page_id"], "regions": regions})
    external_path.write_text(
        "".join(json.dumps(row) + "\n" for row in fixed_external), encoding="utf-8"
    )
    overrides_path.write_text(
        json.dumps(
            {
                "page_id": "p0",
                "region_id": "p0-r1",
                "manual_box": [2, 3, 20, 30],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        anchored_gold,
        "_pdfium_anchor",
        lambda *args, **kwargs: (100.0, 100.0, (10.0, 10.0, 20.0, 20.0), False),
    )

    class FakePage:
        def get_size(self) -> tuple[float, float]:
            return (100.0, 100.0)

    class FakeDocument:
        def __init__(self, _path: str) -> None:
            pass

        def __getitem__(self, _index: int) -> FakePage:
            return FakePage()

    monkeypatch.setitem(
        sys.modules, "pypdfium2", type("FakePdfium", (), {"PdfDocument": FakeDocument})
    )
    anchored_gold.build_anchored_gold(
        selection_path,
        external_path,
        source_dir,
        output_path,
        born_digital_overrides_path=overrides_path,
    )
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["anchoring_method"] == "independent_source_visual_anchor"
    assert records[0]["bounding_box"] == [2.0, 3.0, 20.0, 30.0]
