"""Compact source-grounded semantic extraction prompt."""

from __future__ import annotations

import hashlib
import json

from kawaneen.extraction.contracts import CandidateRegistry

PROMPT_VERSION = "phase11-qwen-prompt-v1"


def render_extraction_prompt(canonical_text: str, registry: CandidateRegistry) -> str:
    candidates = [
        {
            "id": candidate.candidate_id,
            "type": candidate.candidate_type.value,
            "text": candidate.raw_exact_text,
        }
        for candidate in registry.candidates
    ]
    return json.dumps(
        {
            "version": PROMPT_VERSION,
            "instruction": (
                "Select exact source spans only. Use candidate IDs for numeric classifications. "
                "Do not invent metadata or normalized values."
            ),
            "source": canonical_text,
            "candidates": candidates,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def prompt_hash(canonical_text: str, registry: CandidateRegistry) -> str:
    return hashlib.sha256(
        render_extraction_prompt(canonical_text, registry).encode("utf-8")
    ).hexdigest()
