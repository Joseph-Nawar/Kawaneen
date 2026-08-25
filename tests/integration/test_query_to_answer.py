from __future__ import annotations

import pytest
from phase14_support import build_phase14_stack

pytestmark = pytest.mark.integration


def test_query_retrieval_uses_frozen_serving_depths_and_raw_reranker_scores() -> None:
    stack = build_phase14_stack()
    result = stack.retriever.search("الاعتراض خلال ثلاثين يوماً", limit=8)

    assert stack.search_calls == [
        ("الاعتراض خلال ثلاثين يوماً", 50, "sparse"),
        ("الاعتراض خلال ثلاثين يوماً", 50, "dense"),
    ]
    assert result.summary.hit_count == result.summary.returned_count
    assert result.summary.score_type == "reranker_raw_logit"
    assert result.evidence[0].article == "المادة ١٤"
    assert result.evidence[0].provenance in {"sparse-only", "dense-only", "both"}


def test_query_to_grounded_answer_and_abstention_stop_before_generation() -> None:
    stack = build_phase14_stack()

    grounded = stack.answer("الاعتراض خلال ثلاثين يوماً")
    assert grounded.answerable is True
    assert grounded.answer
    assert grounded.citations
    assert grounded.citations[0].document_id == stack.units[0].document_id
    assert grounded.citations[0].document_title == "Synthetic Appeals Regulation"
    assert grounded.citations[0].article == "المادة ١٤"
    assert grounded.citations[0].page == "1"
    assert grounded.citations[0].quoted_text == next(
        unit.text for unit in stack.units if unit.ordinal == 14
    )

    abstained = stack.answer("ما لون السماء؟")
    assert abstained.answerable is False
    assert abstained.answer is None
    assert abstained.citations == ()
