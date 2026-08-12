from __future__ import annotations

from kawaneen.chunking.corpus import freeze_phase5_documents
from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.chunking.strategies import build_chunks
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.normalization.policies import get_policy


def _unit(unit_id: str, ordinal: int, text: str) -> CanonicalUnit:
    return CanonicalUnit(
        unit_id=unit_id,
        document_id="doc-1",
        unit_type=UnitType.FACTS,
        text=text,
        provenance=SourceProvenance(
            source_id="alarb",
            source_version="v1",
            source_path="units.parquet",
            source_row=ordinal,
            source_field="facts",
        ),
        ordinal=ordinal,
    )


def test_fixed_baseline_retains_exact_multi_unit_spans() -> None:
    units = (
        _unit("unit-1", 1, " ".join(f"ألف{index}" for index in range(180))),
        _unit("unit-2", 2, " ".join(f"باء{index}" for index in range(180))),
    )
    corpus = freeze_phase5_documents(units, per_source=1)
    chunks = build_chunks(
        corpus.units, corpus, get_chunk_policy("fixed-256-v1"), get_policy("arabic-light-v1")
    )
    assert chunks
    assert any(len(chunk.source_unit_ids) > 1 for chunk in chunks)
    assert all(chunk.token_count <= 256 for chunk in chunks)
    assert all(span.end > span.start for chunk in chunks for span in chunk.source_spans)


def test_neighbor_context_is_restricted_to_same_parent() -> None:
    units = (
        _unit("unit-1", 1, "ألف نص أول."),
        _unit("unit-2", 2, "باء نص ثان."),
    )
    corpus = freeze_phase5_documents(units, per_source=1)
    chunks = build_chunks(
        corpus.units,
        corpus,
        get_chunk_policy("legal-structure-neighbor-v1"),
        get_policy("arabic-light-v1"),
    )
    assert len(chunks) == 2
    assert all(chunk.context_source_spans for chunk in chunks)
    assert all(
        span.unit_id in {"unit-1", "unit-2"}
        for chunk in chunks
        for span in chunk.context_source_spans
    )
    assert all(chunk.source_spans[0].unit_id in chunk.source_unit_ids for chunk in chunks)


def test_parent_child_strategy_retains_child_citation_and_parent_link() -> None:
    units = (_unit("unit-1", 1, "ألف نص قانوني."),)
    corpus = freeze_phase5_documents(units, per_source=1)
    chunks = build_chunks(
        corpus.units,
        corpus,
        get_chunk_policy("legal-parent-child-v1"),
        get_policy("arabic-light-v1"),
    )
    assert chunks
    assert all(chunk.parent_id for chunk in chunks)
    assert all(chunk.indexed_child_ids for chunk in chunks)
    assert all(chunk.citation_anchor is not None for chunk in chunks)
