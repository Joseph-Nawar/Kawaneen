from __future__ import annotations

import json
from pathlib import Path

import pytest
from private_support import external_review_path, private_repo_path

V5_ITEMS_PARTS = ("phase6_evaluation", "draft-v5", "draft", "selected_and_variants.jsonl")


def _patch() -> Path:
    return external_review_path("phase6_v5_final_literal_patch.jsonl")


def _v5_items() -> Path:
    return private_repo_path(*V5_ITEMS_PARTS)


@pytest.mark.private_artifact
def test_literal_patch_build_applies_all_rows_without_text_reinterpretation() -> None:
    from kawaneen.evaluation.literal_patch import (  # pyright: ignore[reportMissingImports]
        apply_literal_patch,
        load_literal_patch,
    )

    rows = load_literal_patch(_patch())
    result = apply_literal_patch(_v5_items(), _patch())

    assert len(rows) == 240
    assert result.summary.applied_counts == {
        "accept_unchanged": 25,
        "edit_preserve_evidence": 167,
        "replace_with_candidate": 8,
        "variant_rewrite": 40,
    }
    assert result.summary.mismatches == 0
    assert result.summary.evidence_preservation_mismatches == 0
    assert result.summary.replacement_candidate_mismatches == 0
    assert result.summary.variant_parent_mismatches == 0


def test_literal_patch_rejects_unknown_or_duplicate_v5_ids(tmp_path: Path) -> None:
    from kawaneen.evaluation.literal_patch import load_literal_patch

    row = {
        "query_id": "query-original",
        "intent_id": "intent-original",
        "category": "definition",
        "action": "accept_unchanged",
        "old_query_text": "ما القاعدة؟",
        "new_query_text": "ما القاعدة؟",
        "old_gold_answer": "القاعدة",
        "new_gold_answer": "القاعدة",
        "split": "dev",
        "review_provenance": "synthetic",
    }
    row["query_id"] = "query-unknown"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 240 unique patch rows"):
        load_literal_patch(bad, expected_query_ids={"query-known"})


def test_literal_patch_requires_exact_candidate_for_multi_evidence() -> None:
    from kawaneen.evaluation.literal_patch import validate_multi_candidate_spans

    assert validate_multi_candidate_spans((0, 1), (0, 1))
    assert not validate_multi_candidate_spans((0, 2), (0, 1))


@pytest.mark.private_artifact
def test_literal_patch_final_text_and_parent_contract_are_exact() -> None:
    from kawaneen.evaluation.literal_patch import apply_literal_patch, load_literal_patch
    from kawaneen.evaluation.serialization import read_items_jsonl

    result = apply_literal_patch(_v5_items(), _patch())
    rows = {row.query_id: row for row in load_literal_patch(_patch())}
    mapping = {str(row["old_query_id"]): row for row in result.mapping}
    final = {item.query_id: item for item in result.items}
    final_by_intent = {item.intent_id: item for item in result.items if item.variant_id is None}
    candidates = {
        item.query_id: item
        for item in read_items_jsonl(
            private_repo_path("phase6_evaluation", "draft-v5", "draft", "base_candidates.jsonl")
        )
    }
    for old_query_id, row in rows.items():
        item = final[str(mapping[old_query_id]["new_query_id"])]
        if row.action in {"edit_preserve_evidence", "replace_with_candidate"}:
            assert item.query_text == row.new_query_text
            assert item.gold_answer == row.new_gold_answer
        elif row.action == "variant_rewrite":
            assert item.query_text == row.new_query_text
            parent = final_by_intent[item.base_intent_id or ""]
            assert item.gold_answer == parent.gold_answer
            assert item.evidence_groups == parent.evidence_groups
        else:
            assert item.query_text == row.old_query_text
            assert item.gold_answer == row.old_gold_answer
        if row.action == "replace_with_candidate":
            candidate = candidates[str(row.replacement_candidate_query_id)]
            assert item.source_document_ids == candidate.source_document_ids
            assert item.citation_anchors == candidate.citation_anchors
            assert item.intent_id == candidate.intent_id


@pytest.mark.private_artifact
def test_final_candidate_build_and_validator_emit_green_private_candidate() -> None:
    from kawaneen.evaluation.literal_patch import (
        FINAL_PRIVATE_ROOT,
        LiteralPatchSummary,
        _load_snapshot_units,  # pyright: ignore[reportPrivateUsage]
        validate_final_candidate,
    )
    from kawaneen.evaluation.orchestrator import run_build_final_candidate
    from kawaneen.evaluation.serialization import read_items_jsonl

    summary = run_build_final_candidate(patch_file=_patch())
    assert summary["status"] == "phase6_final_candidate_v1_pending_formal_human_review"
    assert summary["validation"]["valid"] is True  # type: ignore[index]
    items = read_items_jsonl(FINAL_PRIVATE_ROOT / "draft" / "selected_and_variants.jsonl")
    units = _load_snapshot_units(FINAL_PRIVATE_ROOT / "corpus" / "canonical_units.json")
    conformance = summary["patch_conformance"]
    conf = LiteralPatchSummary(
        patch_sha256=str(conformance["patch_sha256"]),  # type: ignore[index]
        applied_counts=dict(conformance["applied_counts"]),  # type: ignore[index]
        mismatches=int(conformance["mismatches"]),  # type: ignore[index]
        evidence_preservation_mismatches=int(
            conformance["evidence_preservation_mismatches"]  # type: ignore[index]
        ),
        replacement_candidate_mismatches=int(
            conformance["replacement_candidate_mismatches"]  # type: ignore[index]
        ),
        variant_parent_mismatches=int(conformance["variant_parent_mismatches"]),  # type: ignore[index]
        near_duplicate_pairs=tuple(conformance["near_duplicate_pairs"]),  # type: ignore[index]
    )
    assert (
        validate_final_candidate(
            items,
            {unit_id: unit.text for unit_id, unit in units.items()},
            corpus_hash=str(summary["corpus_hash"]),
            expected_corpus_hash=str(summary["corpus_hash"]),
            conformance=conf,
        )["valid"]
        is True
    )
