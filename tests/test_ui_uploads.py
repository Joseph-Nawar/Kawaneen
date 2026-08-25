import pytest

from kawaneen.ui.uploads import extract_text, segment_text, validate_upload


def test_upload_validation_rejects_unsupported_and_oversized_files() -> None:
    assert validate_upload("brief.docx", 10, 100).accepted is False
    assert validate_upload("brief.txt", 101, 100).reason == "File exceeds the 100-byte limit."
    assert validate_upload("brief.md", 10, 100).accepted is True


def test_text_upload_is_decoded_without_persistence() -> None:
    assert extract_text("brief.txt", "مادة 1".encode()) == "مادة 1"


def test_segment_text_preserves_paragraph_boundaries_and_identity() -> None:
    text = "فقرة أولى.\n\n" + ("فقرة ثانية طويلة. " * 4)

    segments = segment_text(text, max_chars=30, max_segments=5)

    assert len(segments) >= 2
    assert all(len(segment.text) <= 30 for segment in segments)
    assert [segment.segment_id for segment in segments] == [
        f"segment-{index:03d}" for index in range(1, len(segments) + 1)
    ]
    assert "فقرة أولى." in segments[0].text


def test_segment_text_never_silently_truncates() -> None:
    with pytest.raises(ValueError, match="five segments"):
        segment_text("word " * 100, max_chars=10, max_segments=5)


def test_scanned_pdf_explains_that_ocr_is_not_available() -> None:
    with pytest.raises(ValueError, match="OCR"):
        extract_text("scanned.pdf", b"not a real pdf")
