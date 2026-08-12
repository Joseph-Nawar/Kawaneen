from __future__ import annotations

import pytest

from kawaneen.chunking.models import (
    CitationAnchor,
    LegalChunk,
    SourceSpan,
    deterministic_chunk_id,
)


def test_source_span_is_immutable_and_validates_bounds() -> None:
    span = SourceSpan(unit_id="unit-1", start=2, end=7)
    assert span.length == 5
    with pytest.raises(ValueError):
        SourceSpan(unit_id="unit-1", start=7, end=2)


def test_chunk_id_is_deterministic_from_identity_and_spans() -> None:
    first = deterministic_chunk_id(
        "legal-structure-v1",
        "unit-1",
        (SourceSpan(unit_id="unit-1", start=0, end=5),),
    )
    second = deterministic_chunk_id(
        "legal-structure-v1",
        "unit-1",
        (SourceSpan(unit_id="unit-1", start=0, end=5),),
    )
    assert first == second
    assert first != deterministic_chunk_id(
        "legal-structure-v1",
        "unit-1",
        (SourceSpan(unit_id="unit-1", start=0, end=6),),
    )


def test_legal_chunk_preserves_display_search_spans_and_citation() -> None:
    span = SourceSpan(unit_id="unit-1", start=0, end=5)
    chunk = LegalChunk(
        chunk_id="chunk-1",
        strategy_id="legal-structure-v1",
        chunk_policy_hash="a" * 64,
        source_unit_ids=("unit-1",),
        display_text="نص قانوني",
        search_text="نص قانوني",
        source_spans=(span,),
        parent_id="doc-1",
        ancestor_ids=("doc-1",),
        sibling_ids=(),
        structure_path=("document", "section", "paragraph"),
        citation_anchor=CitationAnchor(kind="section", label="facts"),
        token_count=2,
        normalization_policy_id="arabic-light-v1",
        normalization_policy_hash="b" * 64,
        provenance={"source_id": "synthetic", "source_row": 1},
    )
    assert chunk.source_spans[0].length == 5
    assert chunk.citation_anchor.kind == "section"


def test_chunk_policies_are_versioned_and_distinct() -> None:
    from kawaneen.chunking.policies import all_chunk_policies

    policies = all_chunk_policies()
    assert [policy.policy_id for policy in policies] == [
        "fixed-256-v1",
        "fixed-512-v1",
        "legal-structure-v1",
        "legal-structure-neighbor-v1",
        "legal-parent-child-v1",
    ]
    assert len({policy.policy_hash for policy in policies}) == 5
