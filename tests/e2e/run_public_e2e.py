from __future__ import annotations

import time

from kawaneen.ui.client import HttpUiClient
from kawaneen.ui.presentation import inspect_verified_quote


def main() -> None:
    client = HttpUiClient("http://api:8000", timeout=10.0)
    try:
        deadline = time.monotonic() + 30
        while True:
            health = client.health()
            if health.status == "ready":
                break
            if time.monotonic() >= deadline:
                raise AssertionError(f"API did not become ready: {health}")
            time.sleep(0.25)

        search = client.search("ما مهلة الاعتراض؟", limit=8)
        assert search.results
        assert search.results[0].document_id == "phase14-synthetic-appeals-regulation"
        assert search.results[0].article == "المادة ١٢"
        assert search.retrieval.sparse_top_k == 50
        assert search.retrieval.dense_top_k == 50
        assert search.retrieval.fused_candidate_count == 20
        assert search.retrieval.reranker_depth == 8

        answer = client.answer("ما مهلة الاعتراض؟")
        assert answer.answerable is True
        assert answer.answer
        assert len(answer.citations) == 1
        citation = answer.citations[0]
        assert citation.document_id == "phase14-synthetic-appeals-regulation"
        assert citation.article == "المادة ١٢"
        assert citation.page == "1"
        assert citation.quoted_text in answer.answer

        detail = client.get_document(citation.document_id)
        unit = next(item for item in detail.units if citation.quoted_text in item.text)
        rendered = inspect_verified_quote(unit, citation.quoted_text)
        assert "verified-quote" in rendered
        assert citation.quoted_text in rendered

        abstained = client.answer("هل أستطيع رفع دعوى شخصية؟")
        assert abstained.answerable is False
        assert abstained.answer is None
        assert abstained.citations == ()
    finally:
        client.close()


if __name__ == "__main__":
    main()
