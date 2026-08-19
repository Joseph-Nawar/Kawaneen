"""Phase 7 representation adapter over the frozen Phase 4 tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

from kawaneen.normalization.policies import get_policy, normalize_text
from kawaneen.normalization.tokenization import tokenize


@dataclass(frozen=True, slots=True)
class RetrievalRepresentation:
    display_text: str
    search_text: str
    tokens: tuple[str, ...]
    policy_id: str


def tokenize_retrieval(text: str) -> tuple[str, ...]:
    return tokenize(text)


def represent(text: str, policy_id: str) -> RetrievalRepresentation:
    normalized = normalize_text(text, get_policy(policy_id))
    if not isinstance(normalized, str):
        raise TypeError("retrieval normalization unexpectedly returned an audit result")
    return RetrievalRepresentation(
        display_text=text,
        search_text=normalized,
        tokens=tokenize_retrieval(normalized),
        policy_id=policy_id,
    )
