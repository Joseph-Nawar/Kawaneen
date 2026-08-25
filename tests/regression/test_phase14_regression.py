from __future__ import annotations

import pytest
from phase14_support import build_phase14_stack

from regression.conftest import load_cases

pytestmark = pytest.mark.regression


def test_public_regression_cases_exercise_the_complete_synthetic_stack() -> None:
    stack = build_phase14_stack()

    for case in load_cases():
        result = stack.answer(case["query"])
        repeat = stack.answer(case["query"])
        assert tuple(item.chunk_id for item in result.retrieval.evidence) == tuple(
            item.chunk_id for item in repeat.retrieval.evidence
        ), case["id"]
        assert tuple(item.score for item in result.retrieval.evidence) == tuple(
            item.score for item in repeat.retrieval.evidence
        ), case["id"]
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
            if case["top1_article_ordinal"] is not None:
                assert (
                    stack.article_ordinal(result.retrieval.evidence[0].chunk_id)
                    == case["top1_article_ordinal"]
                ), case["id"]
            cited_articles = {
                stack.article_ordinal(
                    next(
                        evidence.contributing_chunk_ids[0]
                        for evidence in stack.context_for(case["query"], result.retrieval).evidence
                        if evidence.evidence_id == citation.evidence_id
                    )
                )
                for citation in result.citations
            }
            assert cited_articles & expected, case["id"]
            assert all(
                any(citation.quoted_text == unit.text for unit in stack.units)
                for citation in result.citations
            ), case["id"]
        else:
            assert result.answerable is False, case["id"]
            assert result.answer is None, case["id"]
            assert result.citations == (), case["id"]


def test_public_regression_cases_are_synthetic_and_exclude_holdout() -> None:
    cases = load_cases()

    assert len(cases) == 20
    assert len({case["id"] for case in cases}) == len(cases)
    assert all("holdout" not in repr(case).lower() for case in cases)
    assert all(case["top_k"] <= 8 for case in cases)
    assert {case["category"] for case in cases} >= {
        "deadline",
        "authority",
        "multiple_articles",
        "insufficient_evidence",
        "abstention",
        "ranking_tie",
    }
