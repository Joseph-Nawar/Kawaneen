# pyright: basic
"""Explicit model adapters; model loading is lazy and never happens at import time."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Encoder = Callable[..., object]
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True, slots=True)
class TokenLengthDiagnostic:
    item_count: int
    p50_tokens: int
    p90_tokens: int
    p95_tokens: int
    p99_tokens: int
    max_tokens: int
    mean_tokens: float
    truncated_count: int
    fraction_above_512: float
    fraction_above_1024: float
    fraction_above_2048: float
    fraction_above_model_maximum: float


@dataclass(frozen=True, slots=True)
class DenseModelAdapter:
    model_id: str
    revision: str = ""
    max_length: int = 512
    default_batch_size: int = 32
    formatting_contract: str = "plain-v1"
    encoder: Encoder | None = None
    embedding_dimension: int = 0
    device: str = "cpu"

    uses_sparse: bool = False
    uses_colbert: bool = False

    def format_query(self, text: str) -> str:
        return text

    def format_passage(self, text: str) -> str:
        return text

    def preload(self) -> None:
        """Load reusable weights without executing an embedding request."""

        if self.encoder is not None:
            return
        from sentence_transformers import SentenceTransformer

        key = (self.model_id, self.revision, self.device)
        model = _MODEL_CACHE.get(key)
        if model is None:
            model = SentenceTransformer(self.model_id, revision=self.revision, device=self.device)
            _MODEL_CACHE[key] = model
        model.max_seq_length = self.max_length

    def _encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        formatted = tuple(texts)
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            key = (self.model_id, self.revision, self.device)
            model = _MODEL_CACHE.get(key)
            if model is None:
                model = SentenceTransformer(
                    self.model_id, revision=self.revision, device=self.device
                )
                _MODEL_CACHE[key] = model
            model.max_seq_length = self.max_length
            raw = model.encode(
                list(formatted),
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            raw = self.encoder(
                formatted,
                batch_size=batch_size,
                max_length=self.max_length,
                normalize=True,
            )
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(formatted):
            raise ValueError("dense encoder returned an invalid matrix shape")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("dense encoder returned NaN or Inf")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("dense encoder returned a zero vector")
        vectors = vectors / norms
        return vectors.astype(np.float32, copy=False)

    def encode_queries(self, texts: Sequence[str], *, batch_size: int = 1) -> np.ndarray:
        return self._encode(tuple(self.format_query(text) for text in texts), batch_size=batch_size)

    def encode_passages(self, texts: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        return self._encode(
            tuple(self.format_passage(text) for text in texts),
            batch_size=batch_size or self.default_batch_size,
        )

    def token_diagnostics(
        self,
        texts: Sequence[str],
        *,
        tokenizer: Callable[..., Any] | None = None,
        already_formatted: bool = False,
    ) -> TokenLengthDiagnostic:
        formatted = (
            tuple(texts)
            if already_formatted
            else tuple(self.format_passage(text) for text in texts)
        )
        lengths: list[int] = []
        if tokenizer is None:
            lengths = [len(text.split()) for text in formatted]
        else:
            for start in range(0, len(formatted), 512):
                encoded = tokenizer(
                    list(formatted[start : start + 512]),
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                )
                lengths.extend(len(ids) for ids in encoded["input_ids"])
        ordered = sorted(lengths)

        def percentile(value: float) -> int:
            return ordered[max(0, int(np.ceil(value * len(ordered))) - 1)] if ordered else 0

        count = len(lengths)

        def fraction_above(limit: int) -> float:
            return sum(length > limit for length in lengths) / max(count, 1)

        return TokenLengthDiagnostic(
            item_count=count,
            p50_tokens=percentile(0.50),
            p90_tokens=percentile(0.90),
            p95_tokens=percentile(0.95),
            p99_tokens=percentile(0.99),
            max_tokens=max(lengths, default=0),
            mean_tokens=sum(lengths) / max(count, 1),
            truncated_count=sum(length > self.max_length for length in lengths),
            fraction_above_512=fraction_above(512),
            fraction_above_1024=fraction_above(1024),
            fraction_above_2048=fraction_above(2048),
            fraction_above_model_maximum=fraction_above(self.max_length),
        )


@dataclass(frozen=True, slots=True)
class E5SmallAdapter(DenseModelAdapter):
    model_id: str = "intfloat/multilingual-e5-small"
    max_length: int = 512
    default_batch_size: int = 32
    formatting_contract: str = "e5-query-passage-v1"
    embedding_dimension: int = 384

    def format_query(self, text: str) -> str:
        return f"query: {text}"

    def format_passage(self, text: str) -> str:
        return f"passage: {text}"


@dataclass(frozen=True, slots=True)
class BGEM3Adapter(DenseModelAdapter):
    model_id: str = "BAAI/bge-m3"
    max_length: int = 1536
    default_batch_size: int = 4
    formatting_contract: str = "bge-m3-dense-v1"
    uses_sparse: bool = False
    uses_colbert: bool = False
    embedding_dimension: int = 1024


def resolve_model_revision(model_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id)
    if not info.sha:
        raise ValueError(f"model revision SHA unavailable for {model_id}")
    return str(info.sha)


def encode_corpus_with_backoff(
    adapter: DenseModelAdapter, texts: Sequence[str]
) -> tuple[np.ndarray, int]:
    batch_size = adapter.default_batch_size
    text_block_size = 4096
    while True:
        try:
            blocks = [
                adapter.encode_passages(
                    texts[start : start + text_block_size], batch_size=batch_size
                )
                for start in range(0, len(texts), text_block_size)
            ]
            return np.vstack(blocks).astype(np.float32, copy=False), batch_size
        except RuntimeError as exc:
            if batch_size <= 1 or "out of memory" not in str(exc).lower():
                raise
            batch_size //= 2


def model_contract_hash(adapter: DenseModelAdapter) -> str:
    payload = {
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "formatting_contract": adapter.formatting_contract,
        "max_length": adapter.max_length,
        "batch_size": adapter.default_batch_size,
        "normalize": True,
        "dtype": "float32",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def loaded_tokenizer(adapter: DenseModelAdapter) -> Callable[..., Any] | None:
    model = _MODEL_CACHE.get((adapter.model_id, adapter.revision, adapter.device))
    return getattr(model, "tokenizer", None) if model is not None else None


def load_tokenizer(adapter: DenseModelAdapter) -> Callable[..., Any]:
    """Load only the locked revision tokenizer, without transferring model weights."""

    key = (adapter.model_id, adapter.revision)
    tokenizer = _TOKENIZER_CACHE.get(key)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(adapter.model_id, revision=adapter.revision)
        _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer
