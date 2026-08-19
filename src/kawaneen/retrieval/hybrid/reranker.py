# pyright: basic
"""Lazy BGE reranker adapter and deterministic candidate ordering."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kawaneen.retrieval.hybrid.contracts import (
    FusedCandidate,
    RerankedCandidate,
    RerankerConfig,
)
from kawaneen.retrieval.models import RetrievalChunk


@dataclass(frozen=True, slots=True)
class PairDiagnostics:
    pair_token_counts: tuple[int, ...]
    truncated_count: int
    max_pair_tokens: int


def _encode(tokenizer: Any, text: str, *, special: bool) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=special)
    return [int(value) for value in encoded]


def _prepare_passage(
    query: str, passage: str, tokenizer: Any, max_length: int
) -> tuple[str, int, bool]:
    query_tokens = _encode(tokenizer, query, special=True)
    passage_tokens = _encode(tokenizer, passage, special=False)
    available = max_length - len(query_tokens)
    if available < 1:
        raise ValueError("query leaves no room for a passage under max_length")
    if len(passage_tokens) <= available:
        return passage, len(query_tokens) + len(passage_tokens), False
    truncated = tokenizer.decode(passage_tokens[:available])
    return str(truncated), max_length, True


def rerank_candidates(
    query: str,
    candidates: Sequence[FusedCandidate],
    chunks: Mapping[str, RetrievalChunk],
    *,
    scorer: Callable[[str, str], float],
    tokenizer: Any,
    config: RerankerConfig,
) -> tuple[tuple[RerankedCandidate, ...], PairDiagnostics]:
    scored: list[RerankedCandidate] = []
    pair_lengths: list[int] = []
    truncated_count = 0
    for candidate in candidates[: config.candidate_count]:
        chunk = chunks.get(candidate.chunk_id)
        if chunk is None:
            raise ValueError(f"reranker candidate is missing from corpus: {candidate.chunk_id}")
        passage, pair_length, truncated = _prepare_passage(
            query, chunk.display_text, tokenizer, config.max_length
        )
        score = float(scorer(query, passage))
        if not math.isfinite(score):
            raise ValueError("reranker scores must be finite")
        scored.append(
            RerankedCandidate(
                chunk_id=candidate.chunk_id,
                score=score,
                prior_fused_rank=candidate.fused_rank,
            )
        )
        pair_lengths.append(pair_length)
        truncated_count += int(truncated)
    scored.sort(key=lambda item: (-item.score, item.prior_fused_rank, item.chunk_id))
    return (
        tuple(scored),
        PairDiagnostics(
            pair_token_counts=tuple(pair_lengths),
            truncated_count=truncated_count,
            max_pair_tokens=max(pair_lengths, default=0),
        ),
    )


class BGERerankerAdapter:
    """Lazy adapter; constructing it never imports or loads model weights."""

    model_id = "BAAI/bge-reranker-v2-m3"

    def __init__(self, *, revision: str, device: str = "cpu", max_length: int = 1024) -> None:
        self.revision = revision
        self.device = device
        self.max_length = max_length
        self._model: Any | None = None

    def _load(self) -> Any:
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("reranker model revision must be a full 40-character SHA")
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_id,
                revision=self.revision,
                max_length=self.max_length,
                device=self.device,
            )
        return self._model

    def score_pairs(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = 4
    ) -> tuple[float, ...]:
        values = self._load().predict(list(pairs), batch_size=batch_size, show_progress_bar=False)
        scores = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("reranker scores must be finite")
        return scores

    def fingerprint(
        self,
        config_hash: str,
        selection_hash: str,
        candidate_chunk_ids: Sequence[str],
        config: RerankerConfig,
        *,
        corpus_hash: str = "",
        chunk_policy_hash: str = "",
        query_id: str = "",
    ) -> str:
        if config.model_id != self.model_id or config.model_revision != self.revision:
            raise ValueError("reranker fingerprint contract does not match the adapter")
        payload = {
            "phase8_config_hash": config_hash,
            "phase7_selection_hash": selection_hash,
            "corpus_hash": corpus_hash,
            "chunk_policy_hash": chunk_policy_hash,
            "query_id": query_id,
            "candidate_chunk_ids": list(candidate_chunk_ids),
            "candidate_fusion_config": {
                "candidate_count": config.candidate_count,
                "evaluation_depth": config.evaluation_depth,
                "serving_depth": config.serving_depth,
            },
            "reranker_model_id": self.model_id,
            "reranker_model_revision": self.revision,
            "max_length": config.max_length,
            "dtype_scoring_contract": config.scoring_contract,
            "device": self.device,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
