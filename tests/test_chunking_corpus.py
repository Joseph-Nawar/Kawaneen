from __future__ import annotations

from kawaneen.chunking.corpus import freeze_phase5_documents
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType


def _unit(source: str, document: str, unit_type: UnitType, row: int) -> CanonicalUnit:
    return CanonicalUnit(
        unit_id=f"{source}-{document}-{unit_type.value}",
        document_id=document,
        unit_type=unit_type,
        text=f"text {row}",
        provenance=SourceProvenance(
            source_id=source,
            source_version="v1",
            source_path="units.parquet",
            source_row=row,
            source_field=unit_type.value,
        ),
    )


def test_freeze_phase5_documents_keeps_whole_document_children() -> None:
    units = tuple(
        unit
        for source in ("alarb", "arabiccr")
        for index in range(3)
        for unit in (
            _unit(source, f"{source}-doc-{index}", UnitType.FACTS, index + 1),
            _unit(source, f"{source}-doc-{index}", UnitType.VERDICT, index + 1),
        )
    )
    corpus = freeze_phase5_documents(units, per_source=2)
    assert corpus.document_count_by_source == {"alarb": 2, "arabiccr": 2}
    assert len(corpus.units) == 8
    assert all(unit.document_id in corpus.document_ids for unit in corpus.units)
    assert corpus == freeze_phase5_documents(units, per_source=2)


def test_freeze_phase5_documents_rejects_unsupported_source() -> None:
    units = (_unit("saudi-moj-derived", "doc", UnitType.ARTICLE, 1),)
    try:
        freeze_phase5_documents(units, per_source=1)
    except ValueError as exc:
        assert "eligible" in str(exc)
    else:
        raise AssertionError("unsupported source was accepted")
