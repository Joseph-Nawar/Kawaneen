"""Deterministic rendering after server-side citation verification."""

from __future__ import annotations

from collections.abc import Sequence

from kawaneen.generation.contracts import ClaimMode, VerifiedClaim


def render_verified_answer(
    claims: Sequence[VerifiedClaim],
    *,
    jurisdiction_text: str | None,
    disclaimer_text: str,
) -> str:
    if not claims:
        raise ValueError("cannot render an answer without verified claims")
    lines: list[str] = []
    if jurisdiction_text and jurisdiction_text.strip():
        lines.append(f"Jurisdiction: {jurisdiction_text}")
    for claim in claims:
        if claim.mode is ClaimMode.INTERPRETATION:
            lines.append(claim.text)
        for citation in claim.citations:
            source = citation.document_title or citation.document_id
            lines.append(f"[{citation.evidence_id}] {citation.quoted_text} ({source})")
    if disclaimer_text.strip():
        lines.append(disclaimer_text)
    return "\n".join(lines)
