from __future__ import annotations

import json
from pathlib import Path

# ruff: noqa: RUF001
from kawaneen.chunking.corpus import freeze_phase5_documents
from kawaneen.chunking.structure import (
    build_structural_leaf_chunks,
    build_structure,
    split_exact_spans,
    validate_structure,
)
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.normalization.policies import get_policy


def _unit(text: str, unit_id: str = "unit-1", document_id: str = "doc-1") -> CanonicalUnit:
    return CanonicalUnit(
        unit_id=unit_id,
        document_id=document_id,
        unit_type=UnitType.FACTS,
        text=text,
        provenance=SourceProvenance(
            source_id="alarb",
            source_version="v1",
            source_path="units.parquet",
            source_row=1,
            source_field="facts",
        ),
        ordinal=1,
    )


def test_split_exact_spans_preserves_text_and_uses_sentence_boundaries() -> None:
    text = "أول فقرة قانونية.\n\nثاني فقرة قانونية."
    spans = split_exact_spans(text, "unit-1", target=4, maximum=8)
    assert spans
    assert all(text[span.start : span.end].strip() for span in spans)
    assert "أول فقرة قانونية." in text[spans[0].start : spans[0].end]
    assert all(span.unit_id == "unit-1" for span in spans)


def test_oversize_sentence_fallback_has_bounded_overlapping_windows() -> None:
    text = " ".join(f"كلمة{index}" for index in range(600))
    spans = split_exact_spans(text, "unit-1", target=384, maximum=512)
    assert len(spans) == 2
    assert spans[0].end > spans[1].start
    assert len(text[spans[0].start : spans[0].end].split()) <= 512
    assert len(text[spans[1].start : spans[1].end].split()) <= 512


def test_oversize_sentence_fallback_is_marked_on_chunks() -> None:
    text = " ".join(f"كلمة{index}" for index in range(600))
    corpus = freeze_phase5_documents((_unit(text),), per_source=1)
    chunks = build_structural_leaf_chunks(corpus.units, corpus, get_policy("arabic-light-v1"))
    assert chunks
    assert all(chunk.fallback_reason == "oversize_fallback" for chunk in chunks)


def test_structure_has_document_section_leaf_without_cross_parent_spans() -> None:
    units = (_unit("فقرة أولى.\n\nفقرة ثانية."),)
    corpus = freeze_phase5_documents(units, per_source=1)
    nodes = build_structure(corpus.units, corpus)
    leaves = build_structural_leaf_chunks(corpus.units, corpus, get_policy("arabic-light-v1"))
    report = validate_structure(nodes, leaves)
    assert report.orphan_count == 0
    assert report.cycle_count == 0
    assert report.cross_parent_boundary_count == 0
    assert len(leaves) >= 1
    assert all(chunk.source_unit_ids == ("unit-1",) for chunk in leaves)


def test_synthetic_article_regressions_remain_independent() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/chunking_legal_structure.json").read_text(
            encoding="utf-8"
        )
    )
    texts = [str(item["text"]) for item in fixture]
    units = tuple(_unit(text, f"unit-{index}", f"doc-{index}") for index, text in enumerate(texts))
    corpus = freeze_phase5_documents(units, per_source=len(units))
    chunks = build_structural_leaf_chunks(units, corpus, get_policy("arabic-light-v1"))
    assert len(chunks) == len(units)
    assert all(
        chunk.source_unit_ids == (unit.unit_id,) for chunk, unit in zip(chunks, units, strict=False)
    )
