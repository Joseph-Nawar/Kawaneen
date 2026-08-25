# Phase 13 Product-Quality Follow-up

This follow-up preserves the Phase 13 Streamlit architecture and visual direction while closing targeted recruiter-facing product gaps. The UI remains evidence-first, consumes only the Phase 12 `/v1` HTTP API, labels synthetic demo data explicitly, and treats browser QA/screenshots as release gates.

## Required improvements

- Search adds only returned-result document filtering and preserves API ranking.
- Ask citation inspection shows the exact verified quote highlighted inside its canonical surrounding unit with metadata, RTL-safe escaped markup, and real source links only when present.
- Extract corpus mode uses paginated `/v1/documents?offset=&limit=` calls and exposes structured findings with segment identity.
- Evaluation becomes a recruiter-grade dashboard with capability readiness, frozen architecture/config, tracked comparison data/deltas, honest generation/extraction cards, separate endpoint-family latency, and collapsed provenance details.
- Custom HTML is escaped and JavaScript-free; `st.html()` is preferred where supported.
- Browser QA uses available local automation and produces synthetic-only screenshots at the four required paths, or records the exact unavailable tooling without fabrication.
- History is rebased onto current `origin/main`, PR #9 is retargeted to `main`, fresh CI is inspected, and the PR remains unmerged.

## Immutable constraints

- Do not access HOLDOUT contents, rerun evaluations, tune models, or modify frozen Phase 7–12 artifacts/results.
- Do not invent confidence scores or unsupported metadata filters.
- Preserve `PHASE11_HYBRID_EXPERIMENTAL_LIMITED` exactly.
- Keep all demo fixtures and screenshots synthetic.
- Push with `--force-with-lease` only after the rebase; do not merge PR #9.
