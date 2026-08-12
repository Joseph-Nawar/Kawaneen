from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kawaneen.corpus.models import SourceProvenance, UnitType
from kawaneen.normalization.corpus import (
    CONTENT_UNIT_TYPES,
    freeze_candidate_policy,
    load_candidate_units,
    select_representative_subset,
)


def _write_units(root: Path, source: str, version: str, rows: list[dict[str, object]]) -> None:
    path = root / source / version / "units.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_candidate_loader_includes_only_content_units_in_stable_order(tmp_path: Path) -> None:
    rows = [
        {
            "unit_id": "b",
            "document_id": "d",
            "unit_type": "facts",
            "text": "content",
            "source_id": "alarb",
            "source_version": "v1",
            "source_path": "source.parquet",
            "source_row": 2,
            "source_field": "facts",
            "split": "",
            "ordinal": None,
        },
        {
            "unit_id": "empty",
            "document_id": "d",
            "unit_type": "facts",
            "text": "  ",
            "source_id": "alarb",
            "source_version": "v1",
            "source_path": "source.parquet",
            "source_row": 3,
            "source_field": "facts",
            "split": "",
            "ordinal": None,
        },
        {
            "unit_id": "metadata",
            "document_id": "d",
            "unit_type": "title",
            "text": "not a candidate",
            "source_id": "alarb",
            "source_version": "v1",
            "source_path": "source.parquet",
            "source_row": 1,
            "source_field": "title",
            "split": "",
            "ordinal": None,
        },
    ]
    _write_units(tmp_path, "alarb", "v1", rows)
    candidates = load_candidate_units(tmp_path, ("alarb",))
    assert [candidate.unit_id for candidate in candidates] == ["b"]
    assert set(CONTENT_UNIT_TYPES) == {
        UnitType.APPLICABLE_LAWS.value,
        UnitType.CASE_TEXT.value,
        UnitType.COURT_REASONING.value,
        UnitType.EVENTS.value,
        UnitType.FACTS.value,
        UnitType.REASONING.value,
        UnitType.RULING.value,
        UnitType.VERDICT.value,
    }


def test_candidate_policy_is_sanitized_and_deterministic() -> None:
    from kawaneen.corpus.models import CanonicalUnit

    units = tuple(
        CanonicalUnit(
            unit_id=unit_id,
            document_id="doc",
            unit_type=UnitType.FACTS,
            text="text",
            provenance=SourceProvenance(
                source_id="alarb",
                source_version="v1",
                source_path="file",
                source_row=index,
                source_field="facts",
            ),
        )
        for index, unit_id in enumerate(("a", "b"), start=1)
    )
    policy = freeze_candidate_policy(units)
    assert policy.candidate_count == 2
    assert policy.included_unit_types == ("facts",)
    assert policy.source_counts == {"alarb": 2}
    assert policy.manifest_hash == freeze_candidate_policy(units).manifest_hash


def test_representative_subset_is_balanced_and_stable() -> None:
    from kawaneen.corpus.models import CanonicalUnit

    units = tuple(
        CanonicalUnit(
            unit_id=f"{source}-{unit_type}-{index:03d}",
            document_id="doc",
            unit_type=UnitType(unit_type),
            text="text",
            provenance=SourceProvenance(
                source_id=source,
                source_version="v1",
                source_path="file",
                source_row=index + 1,
                source_field="text",
            ),
        )
        for source in ("alarb", "arabiccr")
        for unit_type in CONTENT_UNIT_TYPES
        for index in range(5)
    )
    subset = select_representative_subset(units, per_source_unit_type=2)
    assert len(subset) == 32
    assert subset == select_representative_subset(units, per_source_unit_type=2)
