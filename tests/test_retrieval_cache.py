import numpy as np
import pytest

from kawaneen.retrieval.cache import (
    checkpoint_cache_status,
    embedding_cache_fingerprint,
    encode_corpus_checkpointed,
)


def test_embedding_cache_fingerprint_changes_for_every_contract_field() -> None:
    base = dict(
        corpus_hash="corpus",
        policy_hash="policy",
        normalization_policy_hash="norm",
        model_id="model",
        model_revision="revision",
        formatting_contract="query-prefix-v1",
        max_length=512,
        embedding_dimension=384,
        normalize=True,
        dtype="float32",
    )
    original = embedding_cache_fingerprint(**base)

    for key, value in {
        "corpus_hash": "other",
        "policy_hash": "other",
        "normalization_policy_hash": "other",
        "model_id": "other",
        "model_revision": "other",
        "formatting_contract": "other",
        "max_length": 256,
        "embedding_dimension": 768,
        "normalize": False,
        "dtype": "float64",
    }.items():
        changed = {**base, key: value}
        assert embedding_cache_fingerprint(**changed) != original


def test_cached_embeddings_reject_metadata_fingerprint_mismatch(tmp_path) -> None:
    from kawaneen.retrieval.cache import load_cached_embeddings, save_cached_embeddings

    path = tmp_path / "cache"
    vectors = np.eye(2, dtype=np.float32)
    save_cached_embeddings(path, vectors, ("a", "b"), fingerprint="one")
    loaded, ids = load_cached_embeddings(path, fingerprint="one")
    assert np.array_equal(loaded, vectors)
    assert ids == ("a", "b")
    with pytest.raises(ValueError, match="fingerprint"):
        load_cached_embeddings(path, fingerprint="two")


def test_legacy_complete_cache_is_reused_without_encoder_invocation(tmp_path) -> None:
    from kawaneen.retrieval.cache import save_cached_embeddings

    path = tmp_path / "legacy"
    vectors = np.eye(2, dtype=np.float32)
    save_cached_embeddings(path, vectors, ("a", "b"), fingerprint="legacy")

    def fail_if_called(_texts: tuple[str, ...], _batch_size: int) -> np.ndarray:
        raise AssertionError("legacy cache must be reused")

    result = encode_corpus_checkpointed(
        ("a", "b"),
        ("a", "b"),
        path,
        fingerprint="legacy",
        encoder=fail_if_called,
        embedding_dimension=2,
    )

    assert result.cache_status == "legacy_hit"
    assert np.array_equal(result.vectors, vectors)


def _fixture_encoder(calls: list[tuple[tuple[str, ...], int]]):
    def encode(texts: tuple[str, ...], batch_size: int) -> np.ndarray:
        calls.append((texts, batch_size))
        values = np.asarray(
            [[float(ord(text) - 96), 1.0, 0.0] for text in texts],
            dtype=np.float32,
        )
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    return encode


def test_interrupted_checkpoint_encoding_reuses_completed_blocks(tmp_path) -> None:
    import json

    path = tmp_path / "checkpoint"
    texts = ("a", "b", "c", "d", "e")
    chunk_ids = ("id-a", "id-b", "id-c", "id-d", "id-e")
    first_calls: list[tuple[tuple[str, ...], int]] = []
    calls = _fixture_encoder(first_calls)

    def interrupting_encoder(block: tuple[str, ...], batch_size: int) -> np.ndarray:
        if len(first_calls) >= 1:
            raise RuntimeError("interrupted fixture")
        return calls(block, batch_size)

    with pytest.raises(RuntimeError, match="interrupted fixture"):
        encode_corpus_checkpointed(
            texts,
            chunk_ids,
            path,
            fingerprint="fingerprint",
            encoder=interrupting_encoder,
            embedding_dimension=3,
            block_size=2,
            batch_size=7,
        )

    resumed_calls: list[tuple[tuple[str, ...], int]] = []
    result = encode_corpus_checkpointed(
        texts,
        chunk_ids,
        path,
        fingerprint="fingerprint",
        encoder=_fixture_encoder(resumed_calls),
        embedding_dimension=3,
        block_size=2,
        batch_size=7,
    )

    assert [block for block, _ in resumed_calls] == [("c", "d"), ("e",)]
    assert json.loads((path / "chunk_ids" / "block_00000.json").read_text()) == ["id-a", "id-b"]
    assert json.loads((path / "chunk_ids" / "block_00001.json").read_text()) == ["id-c", "id-d"]
    assert json.loads((path / "chunk_ids" / "block_00002.json").read_text()) == ["id-e"]
    assert result.vectors.shape == (5, 3)
    status = checkpoint_cache_status(path, chunk_ids=chunk_ids, fingerprint="fingerprint")
    assert status["completed_blocks"] == 3
    assert status["completed_chunks"] == 5


def test_incomplete_temporary_block_is_ignored(tmp_path) -> None:
    path = tmp_path / "checkpoint"
    texts = ("a", "b", "c", "d")
    ids = ("a", "b", "c", "d")
    encode_corpus_checkpointed(
        texts,
        ids,
        path,
        fingerprint="fingerprint",
        encoder=_fixture_encoder([]),
        embedding_dimension=3,
        block_size=2,
    )
    for name in ("vectors.npy", "ids.json", "metadata.json"):
        (path / name).unlink()
    (path / "blocks" / "block_00001.npy.tmp").write_bytes(b"incomplete")
    (path / "chunk_ids" / "block_00001.json.tmp").write_bytes(b"incomplete")

    calls: list[tuple[tuple[str, ...], int]] = []
    encode_corpus_checkpointed(
        texts,
        ids,
        path,
        fingerprint="fingerprint",
        encoder=_fixture_encoder(calls),
        embedding_dimension=3,
        block_size=2,
    )

    assert calls == []


def test_corrupted_block_hash_recomputes_only_that_block(tmp_path) -> None:
    path = tmp_path / "checkpoint"
    texts = ("a", "b", "c", "d", "e")
    ids = ("a", "b", "c", "d", "e")
    expected = encode_corpus_checkpointed(
        texts,
        ids,
        path,
        fingerprint="fingerprint",
        encoder=_fixture_encoder([]),
        embedding_dimension=3,
        block_size=2,
    ).vectors.copy()
    corrupted = np.load(path / "blocks" / "block_00001.npy", allow_pickle=False)
    corrupted[0, 0] += 100.0
    np.save(path / "blocks" / "block_00001.npy", corrupted, allow_pickle=False)

    calls: list[tuple[tuple[str, ...], int]] = []
    result = encode_corpus_checkpointed(
        texts,
        ids,
        path,
        fingerprint="fingerprint",
        encoder=_fixture_encoder(calls),
        embedding_dimension=3,
        block_size=2,
    )

    assert [block for block, _ in calls] == [("c", "d")]
    assert np.array_equal(result.vectors, expected)


def test_changed_checkpoint_fingerprint_invalidates_cache(tmp_path) -> None:
    path = tmp_path / "checkpoint"
    encode_corpus_checkpointed(
        ("a",),
        ("id-a",),
        path,
        fingerprint="old",
        encoder=_fixture_encoder([]),
        embedding_dimension=3,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        encode_corpus_checkpointed(
            ("a",),
            ("id-a",),
            path,
            fingerprint="new",
            encoder=_fixture_encoder([]),
            embedding_dimension=3,
        )


def test_checkpoint_consolidation_matches_uninterrupted_fixture(tmp_path) -> None:
    texts = ("a", "b", "c", "d", "e")
    ids = ("id-a", "id-b", "id-c", "id-d", "id-e")
    uninterrupted = encode_corpus_checkpointed(
        texts,
        ids,
        tmp_path / "uninterrupted",
        fingerprint="fingerprint",
        encoder=_fixture_encoder([]),
        embedding_dimension=3,
        block_size=2,
    ).vectors
    resumed = encode_corpus_checkpointed(
        texts,
        ids,
        tmp_path / "resumed",
        fingerprint="fingerprint",
        encoder=_fixture_encoder([]),
        embedding_dimension=3,
        block_size=3,
    ).vectors

    assert np.array_equal(resumed, uninterrupted)
