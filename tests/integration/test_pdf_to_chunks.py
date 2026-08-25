from __future__ import annotations

import re
from pathlib import Path

import pytest

from kawaneen.chunking.corpus import Phase5Corpus
from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.chunking.strategies import build_chunks
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.corpus.statutory import parse_article_label
from kawaneen.normalization import get_policy

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parents[1] / "fixtures" / "phase14" / "synthetic_appeals_regulation.pdf"
CORRUPT_FIXTURE = Path(__file__).parents[1] / "fixtures" / "phase14" / "corrupt.pdf"
DOCUMENT_ID = "phase14-synthetic-appeals-regulation"


def _extract_text(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _units_from_pdf(path: Path) -> tuple[CanonicalUnit, ...]:
    lines = [line.strip() for line in _extract_text(path).splitlines() if line.strip()]
    headings = [index for index, line in enumerate(lines) if parse_article_label(line).ordinal]
    units: list[CanonicalUnit] = []
    for ordinal, start in enumerate(headings, start=1):
        end = headings[ordinal] if ordinal < len(headings) else len(lines)
        text = "\n".join(lines[start:end])
        label = parse_article_label(lines[start])
        assert label.ordinal is not None
        units.append(
            CanonicalUnit(
                unit_id=f"{DOCUMENT_ID}:article-{label.ordinal}",
                document_id=DOCUMENT_ID,
                unit_type=UnitType.ARTICLE,
                text=text,
                provenance=SourceProvenance(
                    source_id="phase14-synthetic",
                    source_version="fixture-v1",
                    source_path="tests/fixtures/phase14/synthetic_appeals_regulation.pdf",
                    source_row=ordinal,
                    source_field="article",
                ),
                ordinal=label.ordinal,
            )
        )
    return tuple(units)


def _corpus(units: tuple[CanonicalUnit, ...]) -> Phase5Corpus:
    return Phase5Corpus(
        units=units,
        document_ids=frozenset({DOCUMENT_ID}),
        document_count_by_source={"phase14-synthetic": 1},
        source_versions={"phase14-synthetic": "fixture-v1"},
        document_ids_hash="synthetic-document-hash",
        scope_hash="synthetic-scope-hash",
    )


def test_synthetic_pdf_is_machine_readable_and_has_expected_articles() -> None:
    text = _extract_text(FIXTURE)

    assert "Synthetic Appeals Regulation" in text
    assert "An objection may be submitted within thirty days from notification." in text
    assert "المادة ١٤" in text
    assert not re.search(r"scanned|ocr required", text, re.IGNORECASE)

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


def test_corrupt_pdf_fails_closed() -> None:
    from pypdf.errors import PdfReadError

    with pytest.raises((PdfReadError, ValueError, OSError)):
        _extract_text(CORRUPT_FIXTURE)
