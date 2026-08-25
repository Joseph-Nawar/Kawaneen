"""Evidence-first ranked search page."""

from __future__ import annotations

import streamlit as st

from kawaneen.ui.client import UiApiError
from kawaneen.ui.components import (
    render_evidence_card,
    render_mode_note,
    render_page_intro,
    render_product_header,
    render_status_gate,
    render_warning_list,
)
from kawaneen.ui.presentation import filter_returned_evidence
from kawaneen.ui.state import get_context


def render() -> None:
    client, state = get_context()
    render_product_header(state)
    if not render_status_gate(state):
        return
    render_mode_note(state)
    render_page_intro(
        "Evidence workspace",
        "Search",
        "Ranked legal evidence for Saudi Arabia, with scope and provenance kept visible.",
    )
    with st.form("search_form"):
        query = st.text_input(
            "Search query", placeholder="مثال: ما هي مدة الاعتراض؟", key="search_query"
        )
        left, right = st.columns(2)
        with left:
            st.selectbox("Jurisdiction", ["Saudi Arabia · SA"])
        with right:
            limit = st.slider("Result limit", min_value=1, max_value=8, value=5)
        submitted = st.form_submit_button("Search")
    if submitted:
        if not query.strip():
            st.error("Enter a legal question or search term.")
        else:
            try:
                response = client.search(query, limit)
                state.record_latency("search", response.latency_ms)
                st.session_state["search_response"] = response
                st.session_state["search_query_value"] = query
            except UiApiError as error:
                st.error(error.message)
    response = st.session_state.get("search_response")
    original_query = st.session_state.get("search_query_value", query)
    if response is None:
        st.info(
            "Search returns ranked evidence with exact source metadata. "
            "No answer is generated on this screen."
        )
        return
    st.markdown("### Ranked evidence")
    st.caption(
        f"{response.retrieval.returned_count} results · {response.latency_ms:.0f} ms API latency · "
        "scope: jurisdiction SA"
    )
    with st.expander("Refine returned evidence", expanded=False):
        refinement = st.text_input(
            "Text refinement",
            placeholder="Filter only the returned evidence",
            key="search_refinement",
        )
        document_options = {
            f"{item.document_title or item.document_id} · {item.document_id}": item.document_id
            for item in response.results
        }
        selected_documents = st.multiselect(
            "Filter returned documents",
            options=list(document_options),
            help="Client-side filter derived only from this response; API ranking is preserved.",
        )
    results = filter_returned_evidence(
        response.results,
        {document_options[label] for label in (selected_documents or [])},
    )
    if refinement:
        results = tuple(item for item in results if refinement.casefold() in item.text.casefold())
        st.caption("Refine returned evidence · original ranking preserved · not a new API search")
    if not results:
        st.warning("No returned evidence matches this refinement.")
    for evidence in results:
        render_evidence_card(evidence, original_query)
    with st.expander("Evidence inspector"):
        if results:
            selected = st.selectbox(
                "Inspect result", [f"{item.rank:02d} · {item.document_id}" for item in results]
            )
            item = results[
                [f"{value.rank:02d} · {value.document_id}" for value in results].index(selected)
            ]
            st.write(item.text)
            st.caption(
                f"Article: {item.article or 'not available'} · Page: {item.page or 'not available'}"
            )
        else:
            st.caption("No evidence to inspect.")
    with st.expander("Technical retrieval details"):
        st.caption(
            "Raw reranker logits are technical ranking signals, not calibrated confidence scores."
        )
        for item in results:
            st.write(f"{item.chunk_id} · raw reranker logit {item.score:.4f}")
    render_warning_list(response.warnings)


if __name__ == "__main__":
    render()
