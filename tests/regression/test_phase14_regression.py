from __future__ import annotations

# ruff: noqa: RUF001
import pytest

from kawaneen.retrieval.bm25 import BM25Index
from kawaneen.retrieval.models import RetrievalChunk
from regression.conftest import load_cases

pytestmark = pytest.mark.regression


def _chunk(article: str, text: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=article,
        document_id="phase14-synthetic-appeals-regulation",
        source_id="phase14-synthetic",
        unit_type="article",
        display_text=text,
        search_text=text,
        source_unit_ids=(f"unit-{article}",),
        chunk_policy_hash="c" * 64,
        normalization_policy_id="arabic-light-v1",
        normalization_policy_hash="d" * 64,
        token_count=len(text.split()),
    )


CORPUS = (
    _chunk(
        "article-12",
        "المادة ١٢ Article 12 مهلة الاعتراض objection appeal notification deadline "
        "thirty days from notification ثلاثون يوماً من الإخطار synthetic regulation",
    ),
    _chunk(
        "article-13",
        "المادة ١٣ competent authority acknowledge receipt تستلم الجهة المختصة "
        "الاعتراض إقرار استلام الطلب",
    ),
    _chunk(
        "article-14",
        "المادة ١٤ يجوز تقديم الاعتراض خلال ثلاثين يوماً من تاريخ الإخطار Article 14 objection",
    ),
    _chunk("article-15", "المادة ١٥ قاعدة مشتركة synthetic shared rule"),
    _chunk("article-16", "المادة ١٦ قاعدة مشتركة synthetic shared rule"),
    _chunk("article-17", "المادة ١٧ نص مختلف عن موضوع آخر unrelated provision"),
)


def test_public_regression_cases_have_stable_observable_behavior() -> None:
    index = BM25Index.build(CORPUS, "arabic-light-v1", k1=1.2, b=0.75)

    for case in load_cases():
        hits = index.search(case["query"], top_k=3)
        hit_ids = [hit.chunk_id for hit in hits if hit.score > 0]
        expected_ids = case["expected_chunk_ids"]

        assert isinstance(case["answer"], bool)
        if case["answer"]:
            assert set(expected_ids).issubset(hit_ids), case["id"]
            if case["expected_article"] is not None:
                assert hit_ids[0] == expected_ids[0], case["id"]
        else:
            assert expected_ids == [], case["id"]


def test_regression_cases_are_public_synthetic_and_exclude_holdout() -> None:
    cases = load_cases()

    assert 15 <= len(cases) <= 25
    assert len({case["id"] for case in cases}) == len(cases)
    assert all("holdout" not in repr(case).lower() for case in cases)
    assert {case["category"] for case in cases} >= {
        "deadline",
        "authority",
        "multiple_articles",
        "insufficient_evidence",
        "abstention",
        "ranking_tie",
    }
