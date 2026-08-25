from __future__ import annotations

import math
import os

import pytest
from phase14_support import build_phase14_stack
from regression.conftest import load_cases

from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
from kawaneen.retrieval.serving import HybridServingRetriever

pytestmark = pytest.mark.model_artifact

DENSE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def _missing_cache_files(repo_id: str, revision: str) -> tuple[str, ...]:
    from huggingface_hub import try_to_load_from_cache

    required = ("config.json", "tokenizer_config.json")
    alternatives = ("model.safetensors", "pytorch_model.bin")
    missing = [
        filename
        for filename in required
        if try_to_load_from_cache(repo_id, filename, revision=revision)
        in {None, "_CACHED_NO_EXIST"}
    ]
    if not any(
        try_to_load_from_cache(repo_id, filename, revision=revision)
        not in {None, "_CACHED_NO_EXIST"}
        for filename in alternatives
    ):
        missing.append("model.safetensors or pytorch_model.bin")
    return tuple(missing)


def test_real_bge_model_regression_exercises_all_public_cases_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert os.environ["HF_HUB_OFFLINE"] == "1"

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("sentence-transformers is unavailable; no model execution attempted")

    missing_dense = _missing_cache_files("BAAI/bge-m3", DENSE_REVISION)
    if missing_dense:
        pytest.skip(
            "BAAI/bge-m3 frozen cache is incomplete (missing "
            + ", ".join(missing_dense)
            + "); no download attempted"
        )
    missing_reranker = _missing_cache_files("BAAI/bge-reranker-v2-m3", RERANKER_REVISION)
    if missing_reranker:
        pytest.skip(
            "BAAI/bge-reranker-v2-m3 frozen cache is incomplete (missing "
            + ", ".join(missing_reranker)
            + "); no download attempted"
        )

    from kawaneen.retrieval.dense_models import BGEM3Adapter
    from kawaneen.retrieval.hybrid.reranker import BGERerankerAdapter

    stack = build_phase14_stack()
    dense = BGEM3Adapter(revision=DENSE_REVISION)
    passage_vectors = dense.encode_passages(tuple(chunk.display_text for chunk in stack.chunks))
    assert passage_vectors.dtype.name == "float32"
    assert passage_vectors.shape[0] == len(stack.chunks)
    dense_index = stack.vector_index.__class__.build(
        passage_vectors, [chunk.chunk_id for chunk in stack.chunks]
    )

    reranker = BGERerankerAdapter(revision=RERANKER_REVISION)
    reranker.preload()

    def dense_search(query: str, top_k: int) -> tuple[SourceHit, ...]:
        query_vector = dense.encode_queries((query,))[0]
        return tuple(
            SourceHit(hit.chunk_id, hit.score)
            for hit in dense_index.search(query_vector, top_k=top_k)
        )

    def rerank(query: str, candidates: tuple[object, ...]) -> dict[str, float]:
        pairs = [
            (query, stack.chunks_by_id[candidate.chunk_id].display_text) for candidate in candidates
        ]
        scores = reranker.score_pairs(pairs)
        assert all(math.isfinite(score) for score in scores)
        return {
            candidate.chunk_id: score for candidate, score in zip(candidates, scores, strict=True)
        }

    model_retriever = HybridServingRetriever(
        chunks=stack.chunks_by_id,
        sparse_search=lambda query, top_k: tuple(
            SourceHit(hit.chunk_id, hit.score)
            for hit in stack.bm25.search(query, top_k=top_k)
            if hit.score > 0.0
        ),
        dense_search=dense_search,
        reranker=rerank,
        fusion_config=FusionConfig(sparse_weight=1.0, dense_weight=0.25),
    )

    for case in load_cases():
        first = model_retriever.search(case["query"])
        second = model_retriever.search(case["query"])
        first_signature = tuple((item.chunk_id, item.score) for item in first.evidence)
        second_signature = tuple((item.chunk_id, item.score) for item in second.evidence)
        assert first_signature == second_signature, case["id"]
        assert all(math.isfinite(item.score) for item in first.evidence), case["id"]
        observed = {
            stack.article_ordinal(item.chunk_id) for item in first.evidence[: case["top_k"]]
        }
        if case["answer"]:
            assert set(case["expected_article_ordinals"]) <= observed, case["id"]
            if case["top1_article_ordinal"] is not None:
                assert (
                    stack.article_ordinal(first.evidence[0].chunk_id)
                    == case["top1_article_ordinal"]
                ), case["id"]
