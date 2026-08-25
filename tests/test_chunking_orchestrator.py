from __future__ import annotations

import json

import pytest

from kawaneen.chunking.corpus import freeze_phase5_documents
from kawaneen.chunking.models import CitationAnchor, LegalChunk, SourceSpan
from kawaneen.chunking.orchestrator import (
    _chunk_dict,
    _corpus_manifest,
    _phase3_canonical_hashes,
    _span_dict,
    _write_private_chunks,
    chunking_plan,
    select_chunk_strategy,
    validate_phase5_chunking,
)
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType


def _unit() -> CanonicalUnit:
    return CanonicalUnit(
        unit_id="unit-1",
        document_id="doc-1",
        unit_type=UnitType.FACTS,
        text="نص قانوني.",
        provenance=SourceProvenance(
            source_id="alarb",
            source_version="v1",
            source_path="units.parquet",
            source_row=1,
            source_field="facts",
        ),
        ordinal=1,
    )


def _chunk() -> LegalChunk:
    return LegalChunk(
        chunk_id="chunk-1",
        strategy_id="legal-structure-v1",
        chunk_policy_hash="a" * 64,
        source_unit_ids=("unit-1",),
        display_text="نص قانوني.",
        search_text="نص قانوني.",
        source_spans=(SourceSpan("unit-1", 0, 11),),
        parent_id="parent-1",
        ancestor_ids=("doc-1", "parent-1"),
        sibling_ids=(),
        structure_path=("document", "section", "paragraph"),
        citation_anchor=CitationAnchor(kind="section", source_unit_id="unit-1"),
        token_count=3,
        normalization_policy_id="arabic-light-v1",
        normalization_policy_hash="b" * 64,
        provenance={"source_id": "synthetic"},
    )


def test_chunking_plan_and_sanitized_helpers(tmp_path) -> None:
    plan = chunking_plan()
    assert plan["normalization_policy_id"] == "arabic-light-v1"
    assert len(plan["strategies"]) == 5
    span = SourceSpan("unit-1", 0, 3)
    assert _span_dict(span) == {"unit_id": "unit-1", "start": 0, "end": 3}
    chunk = _chunk()
    serialized = _chunk_dict(chunk)
    assert serialized["chunk_id"] == "chunk-1"
    output_root = tmp_path / "private" / "chunks"
    _write_private_chunks(output_root, chunk.strategy_id, (chunk,))
    lines = (
        (output_root / chunk.strategy_id / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert json.loads(lines[0])["source_spans"] == [{"end": 11, "start": 0, "unit_id": "unit-1"}]


def test_corpus_manifest_preserves_canonical_hashes() -> None:
    corpus = freeze_phase5_documents((_unit(),), per_source=1)
    manifest = _corpus_manifest(corpus, {"canonical.parquet": "c" * 64})
    assert manifest["canonical_hashes"] == {"canonical.parquet": "c" * 64}
    assert manifest["ocr_included"] is False


@pytest.mark.private_artifact
def test_phase3_canonical_hash_inventory_is_read_only_and_complete() -> None:
    hashes = _phase3_canonical_hashes()
    assert len(hashes) == 8
    assert all(len(digest) == 64 for digest in hashes.values())


def test_validate_phase5_artifacts_and_rejection_branch() -> None:
    result = validate_phase5_chunking()
    assert result["valid"] is True
    strategies = {"fixed-256-v1": {"mrr_at_10": 0.1}}
    citation = {"fixed-256-v1": {}}
    context = {"fixed-256-v1": {}}
    gates = {"fixed-256-v1": {"eligible": False}}
    decision = select_chunk_strategy(strategies, citation, context, gates)
    assert decision["selected_policy_id"] is None
    assert decision["best_fixed_baseline"] is None


def test_selection_can_promote_neighbor_when_context_rule_is_met() -> None:
    strategies = {
        "fixed-256-v1": {"recall_at_10": 0.80, "mrr_at_10": 0.70},
        "legal-structure-v1": {"recall_at_10": 0.79, "mrr_at_10": 0.69},
        "legal-structure-neighbor-v1": {"recall_at_10": 0.78, "mrr_at_10": 0.68},
    }
    citation = {
        "fixed-256-v1": {"citation_precision_at_1": 0.10},
        "legal-structure-v1": {"citation_precision_at_1": 0.20},
        "legal-structure-neighbor-v1": {"citation_precision_at_1": 0.20},
    }
    context = {
        "fixed-256-v1": {"context_coverage_at_5": 0.1},
        "legal-structure-v1": {"context_coverage_at_5": 0.50},
        "legal-structure-neighbor-v1": {"context_coverage_at_5": 0.60},
    }
    gates = {strategy: {"eligible": True} for strategy in strategies}
    decision = select_chunk_strategy(strategies, citation, context, gates)
    assert decision["selected_policy_id"] == "legal-structure-neighbor-v1"


def test_selection_keeps_fixed_baseline_without_structural_citation_gain() -> None:
    strategies = {
        "fixed-256-v1": {"recall_at_10": 0.80, "mrr_at_10": 0.70},
        "legal-structure-v1": {"recall_at_10": 0.79, "mrr_at_10": 0.69},
    }
    citation = {
        "fixed-256-v1": {"citation_precision_at_1": 0.20},
        "legal-structure-v1": {"citation_precision_at_1": 0.21},
    }
    context = {strategy: {"context_coverage_at_5": 0.5} for strategy in strategies}
    gates = {strategy: {"eligible": True} for strategy in strategies}
    decision = select_chunk_strategy(strategies, citation, context, gates)
    assert decision["selected_policy_id"] == "fixed-256-v1"


def test_selection_prefers_structure_when_fixed_retrieval_is_tied_and_citation_improves() -> None:
    strategies = (
        "fixed-256-v1",
        "fixed-512-v1",
        "legal-structure-v1",
        "legal-structure-neighbor-v1",
        "legal-parent-child-v1",
    )
    metrics = {
        strategy: {
            "recall_at_10": 0.80 if strategy.startswith("fixed") else 0.79,
            "mrr_at_10": 0.70 if strategy.startswith("fixed") else 0.69,
        }
        for strategy in strategies
    }
    citation = {
        strategy: {
            "citation_precision_at_1": 0.20 if strategy.startswith("fixed") else 0.30,
            "structural_anchor_accuracy_at_1": 0.30 if strategy.startswith("fixed") else 0.80,
        }
        for strategy in strategies
    }
    context = {strategy: {"context_coverage_at_5": 0.5} for strategy in strategies}
    gates = {strategy: {"eligible": True, "boundary_violation_count": 0} for strategy in strategies}
    assert select_chunk_strategy(metrics, citation, context, gates)["selected_policy_id"] == (
        "legal-structure-v1"
    )
