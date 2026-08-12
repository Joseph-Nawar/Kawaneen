from __future__ import annotations

import pytest

# ruff: noqa: RUF001
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.normalization import get_policy
from kawaneen.normalization.records import NormalizedRecord, validate_record_contract


def unit(text: str = "  أ/١٢  ") -> CanonicalUnit:
    return CanonicalUnit(
        unit_id="unit-1",
        document_id="document-1",
        unit_type=UnitType.CASE_TEXT,
        text=text,
        provenance=SourceProvenance(
            source_id="synthetic",
            source_version="v1",
            source_path="fixture.jsonl",
            source_row=1,
            source_field="text",
        ),
    )


def test_normalized_record_preserves_exact_display_text_and_provenance() -> None:
    source = "\ufeff  أ/١٢  "
    record = NormalizedRecord.from_canonical(unit(source), get_policy("arabic-light-v1"))
    assert record.unit_id == "unit-1"
    assert record.display_text == source
    assert record.search_text == "ا/١٢"
    assert record.provenance.source_row == 1
    assert record.source_text_sha256 != record.search_text_sha256
    validate_record_contract(record, source)


def test_record_contract_rejects_display_text_mutation() -> None:
    record = NormalizedRecord.from_canonical(unit(), get_policy("arabic-raw-v1"))
    with pytest.raises(ValueError, match="display_text"):
        validate_record_contract(record.model_copy(update={"display_text": "changed"}), unit().text)


def test_record_contract_rejects_hash_mismatch() -> None:
    record = NormalizedRecord.from_canonical(unit(), get_policy("arabic-raw-v1"))
    with pytest.raises(ValueError, match="source_text_sha256"):
        validate_record_contract(
            record.model_copy(update={"source_text_sha256": "0" * 64}), unit().text
        )
