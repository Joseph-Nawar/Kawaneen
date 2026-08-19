import pytest

from kawaneen.retrieval.artifacts import assert_text_free_tracked_payload


def test_tracked_payload_rejects_query_and_evidence_text_fields() -> None:
    assert_text_free_tracked_payload({"metrics": {"Recall@1": 0.5}, "sample_count": 2})
    with pytest.raises(ValueError, match="text-bearing"):
        assert_text_free_tracked_payload({"query_text": "forbidden"})
    with pytest.raises(ValueError, match="text-bearing"):
        assert_text_free_tracked_payload({"retrieved_text": "forbidden"})
    with pytest.raises(ValueError, match="text-bearing"):
        assert_text_free_tracked_payload({"nested": [{"display_text": "forbidden"}]})


def test_checkpoint_manifests_and_progress_are_text_free(tmp_path) -> None:
    import json

    import numpy as np

    from kawaneen.retrieval.cache import encode_corpus_checkpointed

    encode_corpus_checkpointed(
        ("private query text must not be serialized", "private evidence text"),
        ("chunk-a", "chunk-b"),
        tmp_path / "checkpoint",
        fingerprint="fingerprint",
        encoder=lambda texts, _batch_size: np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)
        ),
        embedding_dimension=2,
    )
    for path in (
        tmp_path / "checkpoint" / "manifest.json",
        tmp_path / "checkpoint" / "progress.json",
    ):
        serialized = path.read_text(encoding="utf-8")
        assert "private query text" not in serialized
        assert "private evidence text" not in serialized
        assert "query_text" not in serialized
        assert "evidence" not in serialized
        assert json.loads(serialized)
