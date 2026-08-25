from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest
from private_support import external_review_path, private_repo_path

from kawaneen.evaluation.candidates_v3 import (
    V3_VERSION,
    build_v3_candidates,
    load_accepted_unanswerables,
)
from kawaneen.evaluation.corpus import freeze_evaluation_corpus, load_evaluation_units
from kawaneen.evaluation.models import Answerability, QueryCategory

V2_ITEMS_PARTS = ("phase6_evaluation", "draft", "selected_and_variants.jsonl")


@pytest.mark.private_artifact
def test_external_adjudication_preserves_exactly_25_accepted_unanswerables() -> None:
    items = load_accepted_unanswerables(
        private_repo_path(*V2_ITEMS_PARTS),
        external_review_path("phase6_independent_ai_source_review_v2.jsonl"),
    )
    assert len(items) == 25
    assert all(item.category is QueryCategory.UNANSWERABLE for item in items)
    assert all(item.answerability is Answerability.UNANSWERABLE for item in items)
    assert all(item.dataset_version == V3_VERSION for item in items)
    assert all(not item.human_verified for item in items)


@pytest.mark.private_artifact
def test_v3_build_restores_quotas_and_generates_contract_preserving_variants(
    tmp_path: Path,
) -> None:
    corpus = freeze_evaluation_corpus(
        load_evaluation_units(Path("data/interim/canonical")),
        canonical_root=Path("data/interim/canonical"),
    )
    pool, selected, variants = build_v3_candidates(
        corpus,
        v2_items_path=private_repo_path(*V2_ITEMS_PARTS),
        adjudication_path=external_review_path("phase6_independent_ai_source_review_v2.jsonl"),
        output_root=tmp_path,
    )
    assert len(pool) == 395
    assert len(selected) == 200
    assert len(variants) == 40
    assert Counter(item.category for item in selected) == {
        QueryCategory.EXACT_PROVISION: 30,
        QueryCategory.DEFINITION: 25,
        QueryCategory.DEADLINE: 20,
        QueryCategory.AUTHORITY: 20,
        QueryCategory.CONDITIONS: 30,
        QueryCategory.MULTI_EVIDENCE: 25,
        QueryCategory.CASE_HOLDING: 25,
        QueryCategory.UNANSWERABLE: 25,
    }
    assert Counter(item.variant_id for item in variants) == {
        "simple-ar": 10,
        "egyptian-ar": 10,
        "english": 10,
        "code-switch": 10,
    }
    base_by_id = {item.intent_id: item for item in selected}
    assert all(
        variant.semantic_target == base_by_id[variant.base_intent_id or ""].semantic_target
        and variant.evidence_groups == base_by_id[variant.base_intent_id or ""].evidence_groups
        and variant.gold_answer == base_by_id[variant.base_intent_id or ""].gold_answer
        for variant in variants
    )
    assert all(not item.human_verified for item in (*selected, *variants))
    english = [item for item in variants if item.variant_id == "english"]
    assert all(not re.search(r"[\u0600-\u06ff]", item.query_text) for item in english)
