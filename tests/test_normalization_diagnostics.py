from __future__ import annotations

from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType

# ruff: noqa: RUF001
from kawaneen.normalization import get_policy
from kawaneen.normalization.diagnostics import diagnose_policy


def _unit(unit_id: str, text: str, row: int) -> CanonicalUnit:
    return CanonicalUnit(
        unit_id=unit_id,
        document_id=unit_id,
        unit_type=UnitType.FACTS,
        text=text,
        provenance=SourceProvenance(
            source_id="synthetic",
            source_version="v1",
            source_path="fixture",
            source_row=row,
            source_field="text",
        ),
    )


def test_diagnostics_report_changes_compression_and_collisions_without_text() -> None:
    units = (
        _unit("one", "ألف", 1),
        _unit("two", "الف", 2),
        _unit("three", "مادة ١", 3),
    )
    diagnostics = diagnose_policy(units, get_policy("arabic-light-v1"))
    assert diagnostics.unit_count == 3
    assert diagnostics.character_change_rate > 0
    assert diagnostics.vocabulary_compression_rate > 0
    assert diagnostics.distinct_form_collision_rate > 0
    assert diagnostics.transformation_frequencies["alef_folded"] == 1
    serialized = diagnostics.to_sanitized_dict()
    assert "ألف" not in str(serialized)
    assert "collision_groups" in serialized
