"""Separate local Streamlit UI for Phase 15 human error review."""

from __future__ import annotations

from pathlib import Path

from .contracts import ErrorCategory, ReviewDecision, ReviewOutcome
from .review import ReviewStore, default_review_paths


def main() -> None:
    # Keep Streamlit optional for public CI and do not import it at package import time.
    import streamlit as st

    st.set_page_config(page_title="Kawaneen Phase 15 review", layout="wide")
    packet_path, progress_path, _manifest_path = default_review_paths(Path.cwd())
    if not packet_path.is_file():
        st.error(f"Review packet is missing: {packet_path}")
        st.stop()
    store = ReviewStore(packet_path, progress_path)
    cases = store.cases()
    status = store.status()
    st.title("Phase 15 diagnostic review")
    st.caption(f"Progress: {status['progress']}")

    case_ids = [case.case_id for case in cases]
    selected_id = st.selectbox("Case", case_ids, index=0)
    case = next(item for item in cases if item.case_id == selected_id)
    st.subheader(f"Case {case.case_id}")
    st.write(
        {
            "language": case.language,
            "pipeline_stage": case.pipeline_stage,
            "legal_category": case.legal_category,
            "answerability": case.answerability,
            "severity": case.severity,
        }
    )
    st.markdown("**Query**")
    st.write(case.query_text or "(private query unavailable in packet)")
    st.markdown("**Evidence and diagnostics**")
    st.write(case.evidence_text or "(private evidence unavailable in packet)")
    st.json(case.diagnostics)

    # Collapsed by default to reduce anchoring. The human decision is authoritative.
    with st.expander("AI preclassification (assistance only)", expanded=False):
        st.caption("This suggestion is not ground truth and must not replace human review.")
        st.write(case.ai_suggestion.value if case.ai_suggestion else "No suggestion")

    current = store.decision_for(case.case_id)
    outcome_values = [None, *list(ReviewOutcome)]
    outcome_default = outcome_values.index(current.outcome) if current else 0
    selected_outcome = st.selectbox("Review outcome", outcome_values, index=outcome_default)
    outcome = ReviewOutcome(selected_outcome) if selected_outcome is not None else None

    primary: ErrorCategory | None = None
    secondary: ErrorCategory | None = None
    if outcome is ReviewOutcome.CONFIRMED_FAILURE:
        primary_values = [None, *list(ErrorCategory)]
        primary_default = (
            primary_values.index(current.primary) if current and current.primary else 0
        )
        primary_value = st.selectbox("Primary root cause", primary_values, index=primary_default)
        primary = ErrorCategory(primary_value) if primary_value is not None else None
        secondary_values = [None, *list(ErrorCategory)]
        secondary_default = (
            secondary_values.index(current.secondary)
            if current and current.secondary is not None
            else 0
        )
        secondary_value = st.selectbox(
            "Optional secondary root cause", secondary_values, index=secondary_default
        )
        secondary = ErrorCategory(secondary_value) if secondary_value is not None else None
    else:
        st.caption("Select an outcome before choosing a root-cause category.")

    confidence = st.slider(
        "Confidence", min_value=1, max_value=5, value=current.confidence if current else 3
    )
    note = st.text_area("Optional note", value=current.note if current and current.note else "")
    can_save = outcome is not None and (
        outcome is not ReviewOutcome.CONFIRMED_FAILURE or primary is not None
    )
    if st.button("Save decision", disabled=not can_save) and outcome is not None:
        store.save_decision(
            ReviewDecision(
                case_id=case.case_id,
                outcome=outcome,
                primary=primary,
                secondary=secondary,
                confidence=confidence,
                note=note or None,
            )
        )
        st.success(f"Saved {store.status()['progress']}")


if __name__ == "__main__":
    main()
