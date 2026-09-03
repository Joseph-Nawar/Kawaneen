from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_demo(root: Path, *, count: int = 3) -> None:
    (root / "corpus").mkdir(parents=True)
    rows = []
    for index in range(count):
        rows.append(
            json.dumps(
                {
                    "chunk_id": f"demo-{index}",
                    "document_id": "demo-returns",
                    "source_id": "synthetic-demo",
                    "document_title": "KAWANEEN_DEMO | محتوى اصطناعي",
                    "article": f"بند {index + 1}",
                    "display_text": (
                        "هذا محتوى اصطناعي للعرض فقط؛ ليس تشريعاً حقيقياً، وليس قانوناً سعودياً، "
                        f"وليس نصيحة قانونية. يقرر البند {index + 1} مدة تجريبية قدرها "
                        f"{index + 1} يوماً."
                    ),
                    "search_text": f"مدة تجريبية {index + 1} يوماً",
                    "source_unit_ids": [f"demo-unit-{index}"],
                    "source_spans": [{"unit_id": f"demo-unit-{index}", "start": 0, "end": 1}],
                    "chunk_policy_hash": "a" * 64,
                    "normalization_policy_id": "arabic-raw-v1",
                    "normalization_policy_hash": "b" * 64,
                    "token_count": 4,
                },
                ensure_ascii=False,
            )
        )
    (root / "corpus" / "chunks.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "ids.json").write_text(
        json.dumps([f"demo-{i}" for i in range(count)]), encoding="utf-8"
    )
    vectors = np.zeros((count, 384), dtype=np.float32)
    for index in range(count):
        vectors[index, index] = 1.0
    np.save(root / "vectors.npy", vectors)
    import hashlib

    manifest = {
        "synthetic": True,
        "jurisdiction": "KAWANEEN_DEMO",
        "corpus_sha256": hashlib.sha256(
            (root / "corpus" / "chunks.jsonl").read_bytes()
        ).hexdigest(),
        "embedding_sha256": hashlib.sha256((root / "vectors.npy").read_bytes()).hexdigest(),
        "model_id": "intfloat/multilingual-e5-small",
        "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "formatting_contract": "e5-query-passage-v1",
        "normalization": "l2-float32",
        "embedding_dimension": 384,
        "vector_count": count,
        "redistribution": "Project-created synthetic demonstration data; freely redistributable.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_demo_corpus_loader_validates_synthetic_metadata_and_hashes(tmp_path: Path) -> None:
    from kawaneen.demo.corpus import load_demo_corpus

    root = tmp_path / "demo"
    _write_demo(root)
    corpus = load_demo_corpus(root)
    assert corpus.manifest["synthetic"] is True
    assert corpus.manifest["jurisdiction"] == "KAWANEEN_DEMO"
    assert len(corpus.chunks) == 3


def test_demo_retrieval_returns_exact_corpus_evidence_and_caps_results(tmp_path: Path) -> None:
    from kawaneen.demo.corpus import load_demo_corpus
    from kawaneen.demo.retrieval import DemoRetriever

    root = tmp_path / "demo"
    _write_demo(root)
    corpus = load_demo_corpus(root)
    retriever = DemoRetriever(
        corpus,
        query_encoder=lambda text: np.pad(np.asarray([1, 0], dtype=np.float32), (0, 382)),
    )

    result = retriever.search("مدة", limit=8)

    assert len(result.evidence) <= 5
    assert result.summary.strategy == "demo_retrieval_first"
    assert all(
        item.text in {chunk.display_text for chunk in corpus.chunks.values()}
        for item in result.evidence
    )


def test_demo_retrieval_can_rerank_only_the_top_four_candidates(tmp_path: Path) -> None:
    from kawaneen.demo.corpus import load_demo_corpus
    from kawaneen.demo.retrieval import DemoRetriever

    root = tmp_path / "demo"
    _write_demo(root)

    class Reranker:
        def score_pairs(
            self, pairs: tuple[tuple[str, str], ...], *, batch_size: int
        ) -> tuple[float, ...]:
            assert len(pairs) == 3
            assert batch_size == 1
            return tuple(float(4 - index) for index in range(len(pairs)))

    retriever = DemoRetriever(
        load_demo_corpus(root),
        query_encoder=lambda text: np.pad(np.asarray([1, 0], dtype=np.float32), (0, 382)),
        reranker=Reranker(),  # type: ignore[arg-type]
    )
    result = retriever.search("مدة", limit=5)

    assert result.summary.reranker_depth == 3
    assert result.summary.score_type == "reranker_raw_logit"
    assert all(item.score_type == "reranker_raw_logit" for item in result.evidence)


def test_demo_reranker_keeps_negative_logit_tail_after_reranked_head(tmp_path: Path) -> None:
    from kawaneen.demo.corpus import load_demo_corpus
    from kawaneen.demo.retrieval import DemoRetriever

    root = tmp_path / "demo"
    _write_demo(root, count=8)

    class Reranker:
        def score_pairs(
            self, pairs: tuple[tuple[str, str], ...], *, batch_size: int
        ) -> tuple[float, ...]:
            assert len(pairs) == 4
            return (-4.0, -3.0, -2.0, -1.0)

    result = DemoRetriever(
        load_demo_corpus(root),
        query_encoder=lambda text: np.pad(np.asarray([1, 0], dtype=np.float32), (0, 382)),
        reranker=Reranker(),  # type: ignore[arg-type]
    ).search("مدة", limit=8)

    assert [item.chunk_id for item in result.evidence] == [
        "demo-3",
        "demo-2",
        "demo-1",
        "demo-0",
        "demo-4",
    ]
    assert [item.score_type for item in result.evidence] == [
        "reranker_raw_logit",
        "reranker_raw_logit",
        "reranker_raw_logit",
        "reranker_raw_logit",
        "rrf_score",
    ]
    assert result.summary.score_type == "mixed"


def test_demo_answerer_is_deterministic_and_has_no_generator() -> None:
    from kawaneen.demo.runtime import DemoAnswerer

    result = type(
        "Result",
        (),
        {
            "evidence": (
                type(
                    "Evidence",
                    (),
                    {
                        "text": "هذا مقتطف اصطناعي.",
                        "document_id": "demo-doc",
                        "document_title": "KAWANEEN_DEMO",
                        "article": "بند 1",
                        "page": None,
                        "source_url": None,
                        "chunk_id": "demo-1",
                    },
                )(),
            ),
            "summary": type("Summary", (), {})(),
        },
    )()

    class Retriever:
        def search(self, query: str, limit: int) -> object:
            del query, limit
            return result

    answerer = DemoAnswerer(
        retriever=Retriever(),
    )

    first = answerer.answer("سؤال")
    second = answerer.answer("سؤال")
    assert first == second
    assert first.answer == "هذا مقتطف اصطناعي."
    assert first.citations[0].quoted_text == first.answer
    assert not hasattr(answerer, "generator")


def test_demo_answerer_abstains_without_positive_lexical_support(tmp_path: Path) -> None:
    from kawaneen.demo.corpus import load_demo_corpus
    from kawaneen.demo.retrieval import DemoRetriever
    from kawaneen.demo.runtime import DemoAnswerer

    root = tmp_path / "demo"
    _write_demo(root)
    answerer = DemoAnswerer(
        DemoRetriever(
            load_demo_corpus(root),
            query_encoder=lambda text: np.pad(np.asarray([1, 0], dtype=np.float32), (0, 382)),
        )
    )

    supported = answerer.answer("مدة تجريبية")
    unrelated = answerer.answer("ما هي إجراءات الطقس في المريخ؟")

    assert supported.answerable is True
    assert supported.answer in {item.text for item in supported.retrieval.evidence}
    assert unrelated.answerable is False
    assert unrelated.answer is None
    assert unrelated.abstention_reason == "INSUFFICIENT_DEMO_EVIDENCE"


def test_public_demo_api_uses_demo_jurisdiction_without_sa_synthetic_mismatch(
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    from kawaneen.demo.runtime import create_demo_app

    root = tmp_path / "demo"
    _write_demo(root)

    def encoder(text: str) -> np.ndarray:
        del text
        return np.pad(np.asarray([1, 0], dtype=np.float32), (0, 382))

    with TestClient(
        create_demo_app(root=root, query_encoder=encoder, use_reranker=False)
    ) as client:
        search = client.post(
            "/v1/search",
            json={"query": "مدة تجريبية", "limit": 1},
        )
        answer = client.post("/v1/answer", json={"query": "مدة تجريبية", "jurisdiction": "SA"})
        extraction = client.post(
            "/v1/extract",
            json={
                "text": "يلتزم الطرف بالسداد خلال ثلاثين يوماً.",
                "jurisdiction": "KAWANEEN_DEMO",
                "mode": "deterministic",
            },
        )

    assert search.status_code == 200
    assert search.json()["jurisdiction"] == "KAWANEEN_DEMO"
    assert search.json()["results"][0]["document_id"].startswith("demo-")
    assert answer.status_code == 200
    assert answer.json()["jurisdiction"] == "KAWANEEN_DEMO"
    assert extraction.status_code == 200
    assert extraction.json()["result"]["jurisdiction"] == "KAWANEEN_DEMO"


def test_demo_retriever_owns_one_e5_adapter_and_preloads_it(tmp_path: Path) -> None:
    from kawaneen.demo.corpus import load_demo_corpus
    from kawaneen.demo.retrieval import DemoRetriever

    root = tmp_path / "demo"
    _write_demo(root)
    retriever = DemoRetriever(load_demo_corpus(root))
    calls: list[str] = []
    retriever.dense_adapter = type(
        "Adapter",
        (),
        {"preload": lambda self: calls.append("preload")},
    )()

    retriever.initialize()
    retriever.initialize()

    assert calls == ["preload"]
