from __future__ import annotations

from kawaneen.chunking.corpus import freeze_phase5_documents
from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.chunking.strategies import build_chunks
from kawaneen.chunking.structure import build_structure
from kawaneen.chunking.validation import summarize_chunks, validate_chunks
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.normalization.policies import get_policy


def _units() -> tuple[CanonicalUnit, ...]:
    return tuple(
        CanonicalUnit(
            unit_id=f"unit-{index}",
            document_id="doc-1",
            unit_type=UnitType.FACTS,
            text=" ".join(f"نص{index}-{word}" for word in range(80)),
            provenance=SourceProvenance(
                source_id="alarb",
                source_version="v1",
                source_path="fixture",
                source_row=index,
                source_field="facts",
            ),
            ordinal=index,
        )
        for index in (1, 2)
    )


def test_chunk_validation_reports_integrity_and_source_coverage() -> None:
    units = _units()
    corpus = freeze_phase5_documents(units, per_source=1)
    nodes = build_structure(units, corpus)
    chunks = build_chunks(
        units, corpus, get_chunk_policy("legal-structure-v1"), get_policy("arabic-light-v1")
    )
    report = validate_chunks(chunks, units, nodes)
    assert report.orphan_count == 0
    assert report.cycle_count == 0
    assert report.invalid_span_count == 0
    assert report.display_text_mismatch_count == 0
    assert report.boundary_violation_count == 0
    assert report.source_coverage_rate == 1.0


def test_chunk_summary_has_required_distribution_fields() -> None:
    units = _units()
    corpus = freeze_phase5_documents(units, per_source=1)
    chunks = build_chunks(
        units, corpus, get_chunk_policy("fixed-256-v1"), get_policy("arabic-light-v1")
    )
    summary = summarize_chunks(chunks, units)
    assert summary["chunk_count"] == len(chunks)
    assert summary["token_mean"] > 0
    assert summary["token_median"] > 0
    assert summary["token_p95"] >= summary["token_median"]
    assert summary["token_max"] <= 256
    assert "duplication_factor" in summary
    assert "fallback_count" in summary
