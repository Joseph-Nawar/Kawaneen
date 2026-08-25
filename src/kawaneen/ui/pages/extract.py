"""Bounded document extraction page with source identity and safe exports."""

from __future__ import annotations

import streamlit as st

from kawaneen.api.contracts import ExtractionResponse
from kawaneen.ui.client import UiApiError
from kawaneen.ui.components import (
    render_mode_note,
    render_page_intro,
    render_product_header,
    render_status_gate,
)
from kawaneen.ui.exports import extraction_csv, extraction_json
from kawaneen.ui.presentation import document_page_bounds, extract_presentation_rows
from kawaneen.ui.state import get_context
from kawaneen.ui.uploads import extract_text, segment_text, validate_upload


def render() -> None:
    client, state = get_context()
    render_product_header(state)
    if not render_status_gate(state):
        return
    render_mode_note(state)
    render_page_intro(
        "Document intelligence",
        "Extract",
        "Turn bounded legal text into traceable candidates and normative structure.",
    )
    source_mode = st.selectbox("Source mode", ["Paste text", "Upload document", "Corpus document"])
    source_text = ""
    if source_mode == "Paste text":
        source_text = st.text_area(
            "Source text",
            placeholder="Paste a provision or clause here.",
            height=140,
            key="source_text",
        )
    elif source_mode == "Upload document":
        uploaded = st.file_uploader("Upload document", type=["txt", "md", "pdf"])
        if uploaded is not None:
            decision = validate_upload(uploaded.name, uploaded.size, 5 * 1024 * 1024)
            if not decision.accepted:
                st.error(decision.reason)
            else:
                try:
                    source_text = extract_text(uploaded.name, uploaded.getvalue())
                    st.caption(
                        f"{uploaded.name} · {len(source_text):,} readable characters · "
                        "no persistence"
                    )
                except ValueError as error:
                    st.error(str(error))
    else:
        try:
            page_limit = 10
            offset = int(st.session_state.get("corpus_document_offset", 0))
            documents = client.list_documents(offset=offset, limit=page_limit)
            start, end, total, has_previous, has_next = document_page_bounds(
                documents.offset, documents.limit, documents.total
            )
            st.markdown("### Corpus documents")
            st.caption(f"Documents {start}\u2013{end} of {total}")
            previous_col, next_col = st.columns(2)
            with previous_col:
                if st.button("Previous documents", disabled=not has_previous):
                    st.session_state["corpus_document_offset"] = max(
                        0, documents.offset - documents.limit
                    )
                    st.rerun()
            with next_col:
                if st.button("Next documents", disabled=not has_next):
                    st.session_state["corpus_document_offset"] = documents.offset + documents.limit
                    st.rerun()
            options = {
                f"{item.title or item.document_id} · {item.document_id}": item.document_id
                for item in documents.items
            }
            selected = st.selectbox("Corpus document", list(options) or ["No documents available"])
            if options and st.button("Load corpus document"):
                detail = client.get_document(options[selected])
                source_text = "\n\n".join(unit.text for unit in detail.units)
                st.session_state["corpus_source_text"] = source_text
            source_text = st.session_state.get("corpus_source_text", source_text)
        except UiApiError as error:
            st.error(error.message)
    mode_label = st.selectbox("Extraction mode", ["Deterministic", "Hybrid"])
    mode = "hybrid" if mode_label == "Hybrid" else "deterministic"
    if st.button("Extract"):
        if not source_text.strip():
            st.error("Provide source text before extracting.")
        else:
            try:
                segments = segment_text(source_text)
                results: list[tuple[str, ExtractionResponse]] = []
                for segment in segments:
                    response = client.extract(segment.text, mode)
                    state.record_latency("extract", response.latency_ms)
                    results.append((segment.segment_id, response))
                st.session_state["extraction_results"] = results
                st.session_state["extraction_mode"] = mode
            except (ValueError, UiApiError) as error:
                st.error(str(error))
    results = st.session_state.get("extraction_results", [])
    if not results:
        st.info(
            "Extraction preserves segment identity and displays provenance rather than "
            "confidence scores."
        )
        return
    if st.session_state.get("extraction_mode") == "hybrid":
        st.warning(
            "PHASE11_HYBRID_EXPERIMENTAL_LIMITED · Hybrid semantic recall remains limited; "
            "review every result."
        )
    st.markdown("### Structured findings")
    for segment_id, response in results:
        result = response.result
        with st.expander(f"{segment_id} · {result.configuration}", expanded=True):
            cards = st.columns(6)
            cards[0].metric("Obligations", len(result.obligations))
            cards[1].metric("Deadlines", len(result.deadlines))
            cards[2].metric("Regulated entities", len(result.regulated_entities))
            cards[3].metric("Exceptions", len(result.exceptions))
            cards[4].metric("Prohibitions", len(result.prohibitions))
            cards[5].metric("Permissions", len(result.permissions))
            rows = extract_presentation_rows(segment_id, response)
            for row in rows:
                field = row["field"]
                value = row["value"]
                if field == "summary" or field == "source":
                    st.json({str(field): value})
                elif field in {"obligation", "rule"}:
                    st.markdown(f"**{str(field).title()}**")
                    st.json(value)
                elif field == "deadline":
                    st.markdown("**Deadline · exact extracted span**")
                    st.json(value)
                elif field in {"regulated_entity", "exception"}:
                    st.markdown(f"**{str(field).replace('_', ' ').title()}**")
                    st.write(value)
    st.download_button(
        "Download JSON", extraction_json(results), "kawaneen-extraction.json", "application/json"
    )
    st.download_button(
        "Download CSV", extraction_csv(results), "kawaneen-extraction.csv", "text/csv"
    )


if __name__ == "__main__":
    render()
