from __future__ import annotations

from pathlib import Path

from kawaneen.evaluation.candidates import build_draft_candidates
from kawaneen.evaluation.corpus import freeze_evaluation_corpus, load_evaluation_units


def test_draft_builder_creates_requested_base_and_variant_counts_without_retrieval_inputs(
    tmp_path: Path,
) -> None:
    corpus = freeze_evaluation_corpus(
        load_evaluation_units(Path("data/interim/canonical")),
        canonical_root=Path("data/interim/canonical"),
    )
    result = build_draft_candidates(corpus, output_root=tmp_path)
    assert len(result.base_candidates) == 360
    assert len(result.selected_base_candidates) == 200
    assert len(result.variants) == 40
    assert all(not item.human_verified for item in result.all_items)
    assert {item.creation_method.value for item in result.all_items} == {
        "document_derived",
        "robustness_variant",
    }


def test_variants_preserve_intent_and_evidence(tmp_path: Path) -> None:
    corpus = freeze_evaluation_corpus(
        load_evaluation_units(Path("data/interim/canonical")),
        canonical_root=Path("data/interim/canonical"),
    )
    result = build_draft_candidates(corpus, output_root=tmp_path)
    base_by_intent = {item.intent_id: item for item in result.selected_base_candidates}
    variants = [item for item in result.variants if item.base_intent_id]
    assert all(item.intent_id in base_by_intent for item in variants)
    assert all(
        item.evidence_groups == base_by_intent[item.intent_id].evidence_groups for item in variants
    )
