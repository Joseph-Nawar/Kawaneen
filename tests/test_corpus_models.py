from __future__ import annotations

import pytest
from pydantic import ValidationError

from kawaneen.corpus.models import (
    CanonicalCase,
    CanonicalStatute,
    SourceFragment,
    SourceProvenance,
    UnitType,
)


def provenance(field: str = "text") -> SourceProvenance:
    return SourceProvenance(
        source_id="fixture",
        source_version="v1",
        source_path="data/file.parquet",
        source_row=1,
        source_field=field,
    )


def test_discriminated_documents_share_provenance_and_preserve_kind() -> None:
    case = CanonicalCase(
        document_id="case-id",
        kind="case",
        title="fixture case",
        provenance=provenance("case_text"),
        split="test",
    )
    statute = CanonicalStatute(
        document_id="statute-id",
        kind="statute",
        title="fixture law",
        provenance=provenance("law_name"),
        reconstruction_status="unique",
    )
    assert case.kind == "case"
    assert statute.kind == "statute"
    assert statute.reconstruction_status == "unique"


def test_source_fragment_requires_exact_source_location() -> None:
    fragment = SourceFragment(
        fragment_id="fragment-id",
        provenance=provenance(),
        raw_label="Article 1",
        derived_article_ordinal=1,
        unit_type=UnitType.ARTICLE_FRAGMENT,
        text="exact fixture text",
    )
    assert fragment.provenance.source_row == 1
    assert fragment.text == "exact fixture text"
    with pytest.raises(ValidationError):
        SourceFragment(
            fragment_id="bad",
            provenance={**provenance().model_dump(), "source_row": 0},
            raw_label="Article 1",
            unit_type=UnitType.ARTICLE_FRAGMENT,
            text="x",
        )
