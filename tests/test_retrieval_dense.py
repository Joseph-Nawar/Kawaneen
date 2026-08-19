from types import SimpleNamespace

import numpy as np
import pytest

import kawaneen.retrieval.dense_models as dense_module
from kawaneen.retrieval.dense_models import (
    BGEM3Adapter,
    E5SmallAdapter,
    encode_corpus_with_backoff,
    load_tokenizer,
    loaded_tokenizer,
    model_contract_hash,
    resolve_model_revision,
)


def test_e5_adapter_formats_query_and_passage_with_prefixes() -> None:
    adapter = E5SmallAdapter(revision="e5-sha")

    assert adapter.format_query("ما الحكم؟") == "query: ما الحكم؟"
    assert adapter.format_passage("النص القانوني") == "passage: النص القانوني"
    assert adapter.max_length == 512
    assert adapter.default_batch_size == 32


def test_bge_adapter_has_no_e5_prefixes_and_uses_dense_only() -> None:
    adapter = BGEM3Adapter(revision="bge-sha")

    assert adapter.format_query("ما الحكم؟") == "ما الحكم؟"
    assert adapter.format_passage("النص القانوني") == "النص القانوني"
    assert adapter.max_length == 1536
    assert adapter.default_batch_size == 4
    assert adapter.uses_sparse is False
    assert adapter.uses_colbert is False


def test_mocked_adapter_returns_finite_float32_normalized_embeddings() -> None:
    adapter = E5SmallAdapter(revision="e5-sha", encoder=lambda texts, **_: np.eye(len(texts), 3))

    vectors = adapter.encode_queries(("a", "b", "c"))

    assert vectors.dtype == np.float32
    assert np.all(np.isfinite(vectors))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_corpus_encoding_halves_only_on_deterministic_oom() -> None:
    calls: list[int] = []

    def encoder(texts, *, batch_size, **_):
        calls.append(batch_size)
        if batch_size > 2:
            raise RuntimeError("out of memory")
        return np.tile(np.asarray([[1.0, 0.0, 0.0]]), (len(texts), 1))

    adapter = E5SmallAdapter(revision="e5-sha", default_batch_size=4, encoder=encoder)
    vectors, batch_size = encode_corpus_with_backoff(adapter, ("a", "b", "c", "d"))

    assert batch_size == 2
    assert vectors.shape == (4, 3)
    assert calls[:2] == [4, 2]


def test_token_length_diagnostics_are_text_free_and_cover_required_thresholds() -> None:
    class Tokenizer:
        def __call__(self, texts, **_):
            return {"input_ids": [list(range(len(text))) for text in texts]}

    adapter = BGEM3Adapter(revision="bge-sha", max_length=4)
    diagnostic = adapter.token_diagnostics(
        ("a", "bb", "ccc", "dddd", "eeeee"),
        tokenizer=Tokenizer(),
        already_formatted=True,
    )

    assert diagnostic.item_count == 5
    assert diagnostic.p50_tokens == 3
    assert diagnostic.p90_tokens == 5
    assert diagnostic.p95_tokens == 5
    assert diagnostic.p99_tokens == 5
    assert diagnostic.max_tokens == 5
    assert diagnostic.fraction_above_512 == 0.0
    assert diagnostic.fraction_above_1024 == 0.0
    assert diagnostic.fraction_above_2048 == 0.0
    assert diagnostic.fraction_above_model_maximum == 0.2


def test_bge_configured_maximum_keeps_observed_corpus_lengths_untruncated() -> None:
    class Tokenizer:
        def __call__(self, texts, **_):
            return {"input_ids": [list(range(len(text))) for text in texts]}

    diagnostic = BGEM3Adapter(revision="bge-sha").token_diagnostics(
        ("x" * 1212,), tokenizer=Tokenizer(), already_formatted=True
    )

    assert diagnostic.truncated_count == 0
    assert diagnostic.fraction_above_model_maximum == 0.0


@pytest.mark.parametrize(
    ("encoder", "message"),
    [
        (lambda _texts, **_kwargs: np.asarray([1.0, 0.0]), "shape"),
        (lambda texts, **_kwargs: np.full((len(texts), 2), np.nan), "NaN"),
        (lambda texts, **_kwargs: np.zeros((len(texts), 2)), "zero"),
    ],
)
def test_adapter_rejects_invalid_mock_encoder_outputs(encoder, message: str) -> None:
    adapter = E5SmallAdapter(revision="e5-sha", encoder=encoder)
    with pytest.raises(ValueError, match=message):
        adapter.encode_queries(("query",))


def test_backoff_does_not_hide_non_memory_runtime_errors() -> None:
    def encoder(_texts, **_kwargs):
        raise RuntimeError("backend failed")

    adapter = E5SmallAdapter(revision="e5-sha", encoder=encoder)
    with pytest.raises(RuntimeError, match="backend failed"):
        encode_corpus_with_backoff(adapter, ("query",))


def test_dense_contract_hash_and_loaded_tokenizer_cache() -> None:
    adapter = E5SmallAdapter(revision="e5-sha")
    key = (adapter.model_id, adapter.revision, adapter.device)
    tokenizer = object()
    dense_module._MODEL_CACHE[key] = type("Model", (), {"tokenizer": tokenizer})()
    try:
        assert model_contract_hash(adapter)
        assert loaded_tokenizer(adapter) is tokenizer
    finally:
        dense_module._MODEL_CACHE.pop(key, None)


def test_locked_revision_and_tokenizer_resolution_are_mockable(monkeypatch) -> None:
    import huggingface_hub
    import transformers

    class Api:
        def model_info(self, _model_id):
            return SimpleNamespace(sha="locked-sha")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert resolve_model_revision("test-model") == "locked-sha"

    tokenizer = object()
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: tokenizer),
    )
    adapter = E5SmallAdapter(revision="tokenizer-sha")
    dense_module._TOKENIZER_CACHE.pop((adapter.model_id, adapter.revision), None)
    assert load_tokenizer(adapter) is tokenizer
