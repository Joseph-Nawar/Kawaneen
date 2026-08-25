"""Reusable Streamlit components for evidence, status, and safe warnings."""

from __future__ import annotations

import html
from collections.abc import Iterable

import streamlit as st

from kawaneen.api.contracts import Citation, Evidence
from kawaneen.ui.config import UiMode
from kawaneen.ui.formatting import highlight_literal, text_direction
from kawaneen.ui.state import UiSessionState, activate_demo_mode
from kawaneen.ui.styles import inject_css


def render_product_header(state: UiSessionState) -> None:
    inject_css()
    status_class = (
        "demo"
        if state.status_label == "Demo data"
        else "degraded"
        if state.status_label == "Degraded"
        else ""
    )
    st.html(
        f"""
        <div class="kw-product-header"
          style="display:flex;justify-content:space-between;align-items:flex-start;
          border-bottom:1px solid #dfe5e3;padding-bottom:1rem;margin-bottom:1.4rem"
        >
          <div>
            <div class="kw-brand">KAWANEEN | قوانين</div>
            <div class="kw-subtitle">Arabic Legal Intelligence · Saudi Arabia</div>
          </div>
          <div class="kw-status {status_class}">{html.escape(state.status_label)}</div>
        </div>
        """,
    )


def render_page_intro(eyebrow: str, title: str, description: str) -> None:
    st.html(f'<div class="kw-eyebrow">{html.escape(eyebrow)}</div>')
    st.title(title)
    st.caption(description)


def render_status_gate(state: UiSessionState) -> bool:
    if state.active_mode is not None:
        return True
    st.warning("The live Phase 12 API is not ready. No demo results are being shown.")
    if state.resolution.requires_demo_activation and st.button("Enter portfolio demo mode"):
        activate_demo_mode()
        st.rerun()
    return False


def render_warning_list(warnings: Iterable[str]) -> None:
    for warning in warnings:
        st.warning(str(warning))


def render_evidence_card(evidence: Evidence, query: str = "") -> None:
    title = html.escape(evidence.document_title or evidence.document_id)
    metadata = " · ".join(
        value
        for value in (evidence.article, f"p. {evidence.page}" if evidence.page else None)
        if value
    )
    excerpt = highlight_literal(evidence.text, query)
    direction = text_direction(evidence.text)
    footer = (
        f"Evidence {html.escape(evidence.chunk_id)} · "
        f"{html.escape(evidence.provenance or 'source')}"
    )
    st.caption(f"{evidence.rank:02d} · {evidence.document_title or evidence.document_id}")
    st.html(
        f"""
        <div class="kw-surface" style="margin:.6rem 0">
          <div style="display:flex;justify-content:space-between;gap:1rem">
            <strong>{evidence.rank:02d} · {title}</strong>
            <span class="kw-meta">{html.escape(metadata)}</span>
          </div>
          <div class="kw-{direction}" style="margin-top:.65rem;line-height:1.8">{excerpt}</div>
          <div class="kw-meta" style="margin-top:.55rem">
            {footer}
          </div>
        </div>
        """,
    )


def render_citation_card(citation: Citation) -> None:
    title = html.escape(citation.document_title or citation.document_id)
    quote = html.escape(citation.quoted_text)
    direction = text_direction(citation.quoted_text)
    st.html(
        f"""
        <div class="kw-quote kw-{direction}" style="margin:.5rem 0">
          <div style="font-weight:700">{title}</div>
          <div style="margin-top:.35rem">“{quote}”</div>
          <div class="kw-meta" style="margin-top:.35rem">
            {html.escape(citation.article or "")} {html.escape(citation.page or "")}
          </div>
        </div>
        """,
    )


def render_mode_note(state: UiSessionState) -> None:
    if state.active_mode is UiMode.DEMO:
        st.info(
            "DEMO DATA · Synthetic Arabic and English fixtures are active; "
            "no live API results are being represented."
        )
