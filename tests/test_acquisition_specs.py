from __future__ import annotations

from pathlib import Path

import pytest

from kawaneen.acquisition.models import AcquisitionPurpose, SourceSpecification
from kawaneen.acquisition.specs import load_specification


def test_phase_2_specs_load_with_pinned_source_facts() -> None:
    alarb = load_specification(Path("data/manifests/acquisition_specs/alarb.toml"))
    arabiccr = load_specification(Path("data/manifests/acquisition_specs/arabiccr.toml"))
    moj = load_specification(Path("data/manifests/acquisition_specs/saudi-moj-derived.toml"))

    assert alarb.source_id == "alarb"
    assert alarb.revision == "e64bfdc867146294a65434c5ca16c2c4c5288ca2"
    assert alarb.expected_records == 13341
    assert alarb.expected_splits == {"train": 12012, "test": 1329}
    assert arabiccr.source_id == "arabiccr"
    assert arabiccr.version == "3"
    assert arabiccr.expected_records == 12806
    assert arabiccr.canonical_source.startswith("Mendeley Data DOI")
    assert "manual official download" in arabiccr.acquisition_method
    assert AcquisitionPurpose.LOCAL_RESEARCH in arabiccr.allowed_purposes
    assert moj.revision == "8b55ef5a666ad773c81086051813582fd14eb466"
    assert moj.expected_records == 3185
    assert moj.files[-1].expected_columns == (
        "text",
        "article_number",
        "law_name",
        "law_type",
        "source",
    )


def test_spec_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        """schema_version = 1
source_id = 'alarb'
unknown_field = 'nope'
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown_field"):
        load_specification(path)


def test_source_spec_requires_relative_expected_files() -> None:
    with pytest.raises(ValueError, match="relative"):
        SourceSpecification.model_validate(
            {
                "schema_version": 1,
                "source_id": "alarb",
                "version": "x",
                "revision": "x",
                "provider": "test",
                "identifier": "test",
                "licence": "Apache-2.0",
                "expected_records": 1,
                "expected_splits": {},
                "allowed_purposes": ["evaluation"],
                "files": [{"path": "/escape.parquet", "format": "parquet"}],
            }
        )
