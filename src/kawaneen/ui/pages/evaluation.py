# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Evidence-backed evaluation dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from kawaneen.ui.components import (
    render_mode_note,
    render_page_intro,
    render_product_header,
    render_status_gate,
)
from kawaneen.ui.evaluation import aggregate_latency, build_evaluation_snapshot
from kawaneen.ui.state import get_context


def render() -> None:
    client, state = get_context()
    render_product_header(state)
    if not render_status_gate(state):
        return
    render_mode_note(state)
    render_page_intro(
        "Evidence and readiness",
        "Evaluation",
        "Tracked metrics, live readiness, and session observations in one transparent view.",
    )
    root = Path(__file__).resolve().parents[4]
    snapshot = build_evaluation_snapshot(root)
    st.markdown("### Live API readiness")
    try:
        models = client.models()
        cols = st.columns(max(1, min(4, len(models.capabilities))))
        for column, capability in zip(cols, models.capabilities, strict=False):
            with column:
                st.metric(capability.capability, "Ready" if capability.ready else "Not ready")
                st.caption(f"{capability.provider} · {capability.model or 'model not reported'}")
    except Exception:
        st.warning("Model readiness could not be loaded from the current API session.")
    st.markdown("### Retrieval")
    st.caption("Frozen Phase 8 architecture/configuration. DEV and holdout remain separate labels.")
    retrieval_rows: list[dict[str, object]] = []
    for split, values in snapshot.retrieval.items():
        if split in {"dev", "holdout"}:
            for model, metrics in values.items():
                retrieval_rows.append({"split": split, "model": model, **metrics})
    st.dataframe(retrieval_rows, width="stretch", hide_index=True)
    st.markdown("### Generation · DEV / independently AI-reviewed, not human-gold")
    generation_cols = st.columns(5)
    labels = (
        ("Valid citations", "ValidCitationRate"),
        ("Claim coverage", "ClaimCitationCoverage"),
        ("Final-answer coverage", "final_answer_coverage"),
        ("False-answer rate", "FalseAnswerRate"),
        ("Invalid generation", "invalid_generation_rate"),
    )
    for column, (label, key) in zip(generation_cols, labels, strict=True):
        with column:
            st.metric(label, f"{snapshot.generation[key]:.3f}")
    st.markdown("### Extraction · protected HOLDOUT summary")
    st.caption(snapshot.extraction["scope_note"])
    extraction_cols = st.columns(7)
    extraction_metrics = (
        ("Completion", "completion"),
        ("Micro precision", "micro_precision"),
        ("Micro recall", "micro_recall"),
        ("Micro F1", "micro_f1"),
        ("Schema validity", "schema_validity"),
        ("Provenance", "provenance_completeness"),
        ("Full-rule exact F1", "full_rule_exact_f1"),
    )
    for column, (label, key) in zip(extraction_cols, extraction_metrics, strict=True):
        with column:
            st.metric(label, f"{snapshot.extraction[key]:.3f}")
    st.bar_chart(snapshot.extraction["error_taxonomy"], horizontal=True)
    st.markdown("### Live session latency — not a benchmark")
    values = state.search_latency_ms + state.answer_latency_ms + state.extract_latency_ms
    summary = aggregate_latency(values)
    if summary.count:
        st.metric("Requests", summary.count)
        st.line_chart(list(summary.values))
    else:
        st.info("No live API requests recorded in this session. Demo latency is excluded.")
    st.markdown("### Provenance · source hashes")
    st.caption("Evaluation source hashes · stale tracked data is detectable")
    for source in snapshot.sources:
        st.code(f"{source.path}\nSHA-256 {source.sha256}")


if __name__ == "__main__":
    render()
