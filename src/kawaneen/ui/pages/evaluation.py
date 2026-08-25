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
from kawaneen.ui.evaluation import (
    aggregate_latency_by_operation,
    build_evaluation_snapshot,
    common_retrieval_comparison,
)
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
                revision = (
                    capability.revision[:12] if capability.revision else "revision not reported"
                )
                st.caption(
                    f"{capability.provider} · {capability.model or 'model not reported'} · "
                    f"rev {revision} · {'ready' if capability.ready else 'not ready'}"
                )
    except Exception:
        st.warning("Model readiness could not be loaded from the current API session.")
    st.markdown("### Frozen retrieval architecture")
    st.caption("BM25 + BGE-M3 → weighted RRF (1.0 / 0.25) → BGE reranker → top 8")
    st.markdown("### Retrieval")
    st.caption("Frozen Phase 8 architecture/configuration. DEV and holdout remain separate labels.")
    retrieval_rows = common_retrieval_comparison(snapshot)
    st.dataframe(list(retrieval_rows), width="stretch", hide_index=True)
    if retrieval_rows:
        chart_data: dict[str, list[float]] = {}
        for row in retrieval_rows:
            label = f"{row['split']} · {row['model']} · {row['metric']}"
            value = row["value"]
            if isinstance(value, (int, float)):
                chart_data[label] = [float(value)]
        st.bar_chart(chart_data, horizontal=True)
    st.markdown("### Phase 8 selected improvement deltas")
    st.dataframe(snapshot.retrieval["selected_deltas"], width="stretch", hide_index=True)
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
    st.caption("Endpoint families: Search · Answer · Extract")
    summaries = aggregate_latency_by_operation(
        {
            "Search": state.search_latency_ms,
            "Answer": state.answer_latency_ms,
            "Extract": state.extract_latency_ms,
        }
    )
    latency_rows = [
        {
            "endpoint": operation,
            "requests": summary.count,
            "median_ms": summary.median,
            "p95_ms": summary.p95,
        }
        for operation, summary in summaries.items()
    ]
    if any(summary.count for summary in summaries.values()):
        st.dataframe(latency_rows, width="stretch", hide_index=True)
        st.bar_chart(
            {row["endpoint"]: [float(row["median_ms"])] for row in latency_rows},
            horizontal=True,
        )
    else:
        st.info("No live API requests recorded in this session. Demo latency is excluded.")
    with st.expander("Technical provenance · source hashes"):
        st.caption("Evaluation source hashes · stale tracked data is detectable")
        for source in snapshot.sources:
            st.code(f"{source.path}\nSHA-256 {source.sha256}")


if __name__ == "__main__":
    render()
