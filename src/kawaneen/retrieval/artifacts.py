# pyright: basic
"""Guards for sanitized tracked Phase 7 outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_FORBIDDEN_KEYS = frozenset(
    {
        "query_text",
        "gold_answer",
        "evidence_groups",
        "chunk_qrels",
        "display_text",
        "search_text",
        "retrieved_text",
        "rankings",
        "per_query",
    }
)


def assert_text_free_tracked_payload(payload: object) -> None:
    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key) in _FORBIDDEN_KEYS:
                    raise ValueError(f"tracked Phase 7 payload contains text-bearing field: {key}")
                walk(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for nested in value:
                walk(nested)

    walk(payload)
