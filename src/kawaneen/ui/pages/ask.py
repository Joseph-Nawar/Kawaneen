"""Grounded answer and citation inspection page; intentionally not chat-based."""

from __future__ import annotations

import html

import streamlit as st

from kawaneen.ui.client import UiApiError
from kawaneen.ui.components import (
    render_citation_card,
    render_mode_note,
    render_page_intro,
    render_product_header,
    render_status_gate,
    render_warning_list,
)
from kawaneen.ui.formatting import text_direction
from kawaneen.ui.presentation import inspect_verified_quote
from kawaneen.ui.state import get_context


def render() -> None:
    client, state = get_context()
    render_product_header(state)
    if not render_status_gate(state):
        return
    render_mode_note(state)
    render_page_intro(
        "Grounded answer", "Ask", "A verified answer workspace with citations in the evidence rail."
    )
    with st.form("ask_form"):
        query = st.text_area(
            "Ask a legal question",
            placeholder="Ask about a provision, deadline, or authority.",
            height=110,
            key="ask_question",
        )
        submitted = st.form_submit_button("Ask")
    if submitted:
        if not query.strip():
            st.error("Enter a legal question.")
        else:
            try:
                response = client.answer(query)
                state.record_latency("answer", response.latency_ms)
                st.session_state["answer_response"] = response
            except UiApiError as error:
                st.error(error.message)
    response = st.session_state.get("answer_response")
    answer_col, evidence_col = st.columns([3, 2], gap="large")
    with answer_col:
        st.markdown("### Grounded answer")
        if response is None:
            st.info(
                "The answer area will show only when the API can ground a response "
                "in verified evidence."
            )
        elif response.answerable and response.answer:
            direction = text_direction(response.answer)
            answer_html = (
                f'<div class="kw-surface kw-{direction}" style="line-height:1.9">'
                f"{html.escape(response.answer)}</div>"
            )
            st.markdown(
                answer_html,
                unsafe_allow_html=True,
            )
        else:
            reason = html.escape(
                response.abstention_reason or "The available evidence was insufficient."
            )
            abstention_html = (
                '<div class="kw-surface" style="border-color:#e4bf87;'
                'background:#fff7e8"><strong>Grounded answer not issued</strong>'
                f'<div style="margin-top:.5rem">{reason}</div>'
                '<div class="kw-meta" style="margin-top:.6rem">'
                "Abstention is an intentional safety decision.</div></div>"
            )
            st.markdown(
                abstention_html,
                unsafe_allow_html=True,
            )
    with evidence_col:
        st.markdown("### Evidence rail")
        if response is None:
            st.caption("Verified quotes and source metadata will appear here.")
        else:
            if not response.citations:
                st.caption("No verified citation was issued.")
            for index, citation in enumerate(response.citations, start=1):
                render_citation_card(citation)
                with st.expander(f"Inspect citation {index}"):
                    try:
                        detail = client.get_document(citation.document_id)
                        location_html = ""
                        for unit in detail.units:
                            location_html = inspect_verified_quote(unit, citation.quoted_text)
                            if location_html:
                                st.success(f"Exact quote located in canonical unit {unit.unit_id}.")
                                document_title = (
                                    detail.document.title or detail.document.document_id
                                )
                                article = citation.article or (
                                    unit.heading_path[-1] if unit.heading_path else "not available"
                                )
                                st.caption(
                                    f"Document: {document_title} · "
                                    f"Unit: {unit.unit_id} · Article: "
                                    f"{article} · "
                                    f"Page: {citation.page or 'not available'}"
                                )
                                st.html(location_html)
                                break
                        if not location_html:
                            st.info(
                                "The exact quote was not found in the returned canonical units."
                            )
                        if citation.source_url:
                            st.link_button("Open real source", citation.source_url)
                    except UiApiError as error:
                        st.warning(error.message)
    if response is not None:
        with st.expander("Retrieval details"):
            st.write(
                f"Strategy: {response.retrieval.strategy} · "
                f"returned: {response.retrieval.returned_count}"
            )
        render_warning_list(response.warnings)


if __name__ == "__main__":
    render()
