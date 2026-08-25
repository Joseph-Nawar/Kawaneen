from __future__ import annotations

import pytest
from regression.conftest import load_cases

pytestmark = pytest.mark.regression


def test_all_public_cases_run_through_the_shared_synthetic_stack() -> None:
    from phase14_support import build_phase14_stack

    stack = build_phase14_stack()
    assert len(stack.units) == 3
    assert len(stack.chunks) == 3

    for case in load_cases():
        result = stack.answer(case["query"])
        expected = set(case["expected_article_ordinals"])
        observed = {
            stack.article_ordinal(evidence.chunk_id)
            for evidence in result.retrieval.evidence[: case["top_k"]]
        }
        if case["answer"]:
            assert expected <= observed, case["id"]
            assert result.answerable is True, case["id"]
            assert result.answer, case["id"]
            assert result.citations, case["id"]
        else:
            assert result.answerable is False, case["id"]
            assert result.answer is None, case["id"]
            assert result.citations == (), case["id"]
