"""Frozen experiment tokenizer shared by diagnostics and lexical retrieval."""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text))
