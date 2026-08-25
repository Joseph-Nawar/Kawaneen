from __future__ import annotations

from pathlib import Path

import pytest

from kawaneen.evaluation.corpus import (
    CONTENT_POLICY_VERSION,
    freeze_evaluation_corpus,
    load_evaluation_units,
)


@pytest.mark.private_artifact
def test_full_scope_uses_all_governed_source_documents_not_phase5_subset() -> None:
    units = load_evaluation_units(Path("data/interim/canonical"))
    corpus = freeze_evaluation_corpus(units, canonical_root=Path("data/interim/canonical"))
    assert corpus.document_count_by_source == {"alarb": 13341, "arabiccr": 12806}
    assert corpus.unit_count == 91782
    assert corpus.document_count > 3000
    assert corpus.content_policy_version == CONTENT_POLICY_VERSION


@pytest.mark.private_artifact
def test_corpus_policy_excludes_moj_and_case_text_when_structured_content_exists() -> None:
    units = load_evaluation_units(Path("data/interim/canonical"))
    assert {unit.provenance.source_id for unit in units} == {"alarb", "arabiccr"}
    assert all(unit.provenance.source_id != "saudi-moj-derived" for unit in units)
    assert all(unit.unit_type.value != "case_text" for unit in units)


def test_corpus_freeze_rejects_unknown_source() -> None:
    from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType

    unit = CanonicalUnit(
        unit_id="u",
        document_id="d",
        unit_type=UnitType.FACTS,
        text="نص",
        provenance=SourceProvenance(
            source_id="unknown",
            source_version="1",
            source_path="x",
            source_row=1,
            source_field="x",
        ),
    )
    with pytest.raises(ValueError, match="eligible"):
        freeze_evaluation_corpus((unit,), canonical_root=Path("data/interim/canonical"))
