"""Reusable Streamlit components for evidence, status, and safe warnings."""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import streamlit as st

from kawaneen.api.contracts import Citation, Evidence
from kawaneen.ui.config import UiMode
from kawaneen.ui.formatting import highlight_literal, text_direction
from kawaneen.ui.state import UiSessionState, activate_demo_mode
from kawaneen.ui.styles import inject_css

_PRODUCT_NAVIGATION = (
    ("pages/search.py", "Search"),
    ("pages/ask.py", "Ask"),
    ("pages/extract.py", "Extract"),
    ("pages/evaluation.py", "Evaluation"),
)


def render_product_navigation(active_page: str | None = None) -> None:
    with st.container(key="kw-product-navigation"):
        columns = st.columns(len(_PRODUCT_NAVIGATION), gap="small", vertical_alignment="center")
        for column, (path, label) in zip(columns, _PRODUCT_NAVIGATION, strict=True):
            with column:
                st.page_link(path, label=label, width="stretch")


def render_product_header(state: UiSessionState, active_page: str | None = None) -> None:
    inject_css()
    status_class = (
        "demo"
        if state.status_label == "Demo data"
        else "degraded"
        if state.status_label == "Degraded"
        else ""
    )
    if state.settings.public_demo:
        subtitle = "Arabic Legal Intelligence · Synthetic public demo"
    elif state.active_mode is UiMode.DEMO:
        subtitle = "Arabic Legal Intelligence · Synthetic fixture mode"
    else:
        subtitle = "Arabic Legal Intelligence · Saudi Arabia"
    st.html(
        f"""
        <div class="kw-product-header">
          <div>
            <div class="kw-brand">KAWANEEN | قوانين</div>
            <div class="kw-subtitle">{html.escape(subtitle)}</div>
          </div>
          <div class="kw-status {status_class}">{html.escape(state.status_label)}</div>
        </div>
        """,
    )
    render_product_navigation(active_page)
    if state.settings.public_demo:
        st.html(
            """
            <div class="kw-public-banner" role="note">
              <span class="kw-public-label">PUBLIC DEMO</span>
              <span class="kw-public-facts">Fictional/synthetic curated corpus ·
              Reduced retrieval profile · No generative legal answer · Not real Saudi legislation ·
              Not legal advice</span>
            </div>
            """
        )


def render_page_intro(eyebrow: str, title: str, description: str) -> None:
    st.html(
        f'<section class="kw-page-intro">'
        f'<div class="kw-eyebrow">{html.escape(eyebrow)}</div>'
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="kw-page-description">{html.escape(description)}</p>'
        f"</section>"
    )


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
        f"{html.escape(evidence.provenance or 'source')} · chunk {html.escape(evidence.chunk_id)}"
    )
    st.html(
        f"""
        <article class="kw-evidence-card">
          <div class="kw-evidence-heading">
            <strong><span class="kw-rank">{evidence.rank:02d}</span> · {title}</strong>
            <span class="kw-meta">{html.escape(metadata or "location not reported")}</span>
          </div>
          <div class="kw-evidence-text kw-{direction}">{excerpt}</div>
          <div class="kw-meta kw-evidence-footer">
            {footer}
          </div>
        </article>
        """,
    )


def render_citation_card(citation: Citation, index: int | None = None) -> None:
    title = html.escape(citation.document_title or citation.document_id)
    quote = html.escape(citation.quoted_text)
    direction = text_direction(citation.quoted_text)
    citation_index = f"{index:02d} · " if index else ""
    st.html(
        f"""
        <article class="kw-quote kw-{direction}" style="margin:.5rem 0">
          <div style="font-weight:700">
            <span class="kw-citation-index">{citation_index}</span>{title}
          </div>
          <div style="margin-top:.35rem">“{quote}”</div>
          <div class="kw-meta" style="margin-top:.35rem">
            {html.escape(citation.article or "")} {html.escape(citation.page or "")}
          </div>
        </article>
        """,
    )


def render_findings_summary(counts: Mapping[str, int]) -> None:
    labels = (
        ("Obligations", "obligations"),
        ("Deadlines", "deadlines"),
        ("Regulated entities", "regulated_entities"),
        ("Exceptions", "exceptions"),
        ("Prohibitions", "prohibitions"),
        ("Permissions", "permissions"),
    )
    items = "".join(
        f'<div class="kw-finding-count"><div class="kw-finding-label">{label}</div>'
        f'<div class="kw-finding-number">{int(counts.get(key, 0))}</div></div>'
        for label, key in labels
    )
    st.html(f'<div class="kw-findings-summary" role="list">{items}</div>')


def render_finding_record(field: str, value: object) -> None:
    label = field.replace("_", " ")
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        headline = mapping.get("text") or mapping.get("action") or mapping.get("actor") or ""
        headline_text = html.escape(str(headline))
        details = " · ".join(
            f"{html.escape(str(key))}: {html.escape(str(item))}"
            for key, item in mapping.items()
            if key not in {"text", "action", "actor"} and item not in (None, [], "")
        )
        body = f'<div class="kw-finding-value">{headline_text}</div>'
        if details:
            body += f'<div class="kw-meta" style="margin-top:.3rem">{details}</div>'
    else:
        body = f'<div class="kw-finding-value">{html.escape(str(value))}</div>'
    st.html(
        f'<article class="kw-finding-record"><div class="kw-finding-type">'
        f"{html.escape(label)}</div>{body}</article>"
    )


def render_status_strip(items: Sequence[tuple[str, str]]) -> None:
    content = "".join(
        f'<div class="kw-status-item"><div class="kw-status-key">{html.escape(key)}</div>'
        f'<div class="kw-status-value">{html.escape(value)}</div></div>'
        for key, value in items
    )
    st.html(f'<div class="kw-status-strip" role="status">{content}</div>')


def render_mode_note(state: UiSessionState) -> None:
    if state.settings.public_demo:
        st.html(
            '<div class="kw-mode-note"><strong>Profile limits</strong> · '
            "queries 500 characters · evidence 5 results · extraction 8,000 characters · "
            "one concurrent request</div>"
        )
        return
    if state.active_mode is UiMode.DEMO:
        st.info(
            "DEMO DATA · Synthetic Arabic and English fixtures are active; "
            "no live API results are being represented."
        )
