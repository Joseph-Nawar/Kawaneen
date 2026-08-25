from __future__ import annotations

import pytest

pytestmark = pytest.mark.model_artifact


def _cached(repo_id: str, revision: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return try_to_load_from_cache(repo_id, "config.json", revision=revision) not in {
        None,
        "_CACHED_NO_EXIST",
    }


def test_real_bge_model_regression_uses_only_existing_local_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_revision = "5617a9f61b028005a4858fdac845db406aefb181"
    reranker_revision = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    if not _cached("BAAI/bge-m3", dense_revision):
        pytest.skip(
            "BAAI/bge-m3 frozen revision is not in the local Hugging Face cache; "
            "no download attempted"
        )
    if not _cached("BAAI/bge-reranker-v2-m3", reranker_revision):
        pytest.skip(
            "BAAI/bge-reranker-v2-m3 frozen revision is not in the local Hugging Face cache; "
            "no download attempted"
        )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from kawaneen.retrieval.dense_models import BGEM3Adapter
    from kawaneen.retrieval.hybrid.reranker import BGERerankerAdapter

    try:
        dense = BGEM3Adapter(revision=dense_revision)
        vectors = dense.encode_queries(("ما مهلة الاعتراض؟",))
        assert vectors.dtype.name == "float32"
        assert vectors.shape == (1, 1024)
        assert vectors[0].dot(vectors[0]) == pytest.approx(1.0, abs=1e-4)

        reranker = BGERerankerAdapter(revision=reranker_revision)
        reranker.preload()
        scores = reranker.score_pairs(
            [("ما مهلة الاعتراض؟", "An objection may be submitted within thirty days.")]
        )
        assert len(scores) == 1
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        pytest.skip(f"frozen model caches cannot execute offline: {type(error).__name__}")
