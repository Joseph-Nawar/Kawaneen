from __future__ import annotations

import pytest
from phase14_support import (
    CORRUPT_FIXTURE,
    DOCUMENT_ID,
    FIXTURE,
    build_synthetic_corpus,
    build_synthetic_units,
    extract_synthetic_pdf_text,
)

from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.chunking.strategies import build_chunks
from kawaneen.normalization import get_policy

pytestmark = pytest.mark.integration


def _extract_text(path):
    return extract_synthetic_pdf_text(path)


def _units_from_pdf(path):
    return build_synthetic_units(path)


def _corpus(units):
    return build_synthetic_corpus(units)


def test_synthetic_pdf_is_machine_readable_and_has_expected_articles() -> None:
    text = _extract_text(FIXTURE)

    assert "Synthetic Appeals Regulation" in text
    assert "An objection may be submitted within thirty days from notification." in text
    assert "المادة ١٤" in text
    assert not any(marker in text.casefold() for marker in ("scanned", "ocr required"))

    units = _units_from_pdf(FIXTURE)
    assert [unit.ordinal for unit in units] == [12, 13, 14]
    assert all(unit.document_id == DOCUMENT_ID for unit in units)
    assert "thirty days" in units[0].text


def test_pdf_to_chunks_preserves_article_boundaries_and_deterministic_provenance() -> None:
    units = _units_from_pdf(FIXTURE)
    corpus = _corpus(units)
    policy = get_policy("arabic-light-v1")
    chunk_policy = get_chunk_policy("legal-structure-v1")

    first = build_chunks(units, corpus, chunk_policy, policy)
    second = build_chunks(units, corpus, chunk_policy, policy)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert {unit_id for chunk in first for unit_id in chunk.source_unit_ids} == {
        unit.unit_id for unit in units
    }
    by_id = {unit.unit_id: unit for unit in units}
    for chunk in first:
        assert chunk.provenance["source_id"] == "phase14-synthetic"
        for span in chunk.source_spans:
            assert span.unit_id in by_id
            assert 0 <= span.start < span.end <= len(by_id[span.unit_id].text)

    assert all(
        len(chunk.source_unit_ids) == 1 and by_id[chunk.source_unit_ids[0]].ordinal in {12, 13, 14}
        for chunk in first
    )


def test_parser_boundary_reports_embedded_text_health() -> None:
    from kawaneen.parsing.health import probe_pdf

    pages = probe_pdf(FIXTURE)
    assert len(pages) == 1
    assert pages[0].text_chars > 0
    assert pages[0].image_count == 0


def test_corrupt_pdf_fails_closed() -> None:
    from pypdf.errors import PdfReadError

    with pytest.raises((PdfReadError, ValueError, OSError)):
        _extract_text(CORRUPT_FIXTURE)
