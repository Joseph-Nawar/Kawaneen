"""Stable text rendering for generator context and evidence."""

from __future__ import annotations

from collections.abc import Mapping

from kawaneen.grounding.contracts import ContextPack, EvidenceReference


def _metadata_lines(pack: ContextPack, block_index: int) -> list[str]:
    block = pack.blocks[block_index]
    source = block.source
    lines = [f"Document: {source.document_title or block.document_id}"]
    if source.jurisdiction is not None:
        lines.append(f"Jurisdiction: {source.jurisdiction}")
    if source.article is not None:
        lines.append(f"Article: {source.article}")
    if source.page is not None:
        lines.append(f"Page: {source.page}")
    if source.source_url is not None:
        lines.append(f"Source URL: {source.source_url}")
    return lines


def render_context(
    pack: ContextPack,
    *,
    evidence_labels: Mapping[str, str] | None = None,
) -> str:
    """Render only server-selected context, including local evidence IDs."""

    evidence_by_unit: dict[str, EvidenceReference] = {}
    for item in pack.evidence:
        evidence_by_unit.setdefault(item.unit_id, item)
    lines: list[str] = []
    previous_document: str | None = None
    previous_heading: tuple[str, ...] | None = None
    for index, block in enumerate(pack.blocks):
        if block.document_id != previous_document:
            lines.extend(_metadata_lines(pack, index))
            previous_heading = None
        if block.heading_path != previous_heading and block.heading_path:
            lines.append(f"Heading: {' / '.join(block.heading_path)}")
        for unit in block.units:
            evidence = evidence_by_unit[unit.unit_id]
            label = (
                evidence_labels.get(evidence.evidence_id, evidence.evidence_id)
                if evidence_labels is not None
                else evidence.evidence_id
            )
            lines.append(f"[{label}] {unit.display_text}")
        previous_document = block.document_id
        previous_heading = block.heading_path
    return "\n".join(lines)


def render_evidence(pack: ContextPack) -> str:
    """Render the evidence-only view used by deterministic verifier tests."""

    return "\n".join(f"[{item.evidence_id}] {item.display_text}" for item in pack.evidence)
