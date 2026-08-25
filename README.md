# Kawaneen

Kawaneen is a jurisdiction-aware Arabic legal and regulatory intelligence system. The current release includes the Phase 7–11 frozen retrieval, grounding, generation-policy, and extraction primitives plus the Phase 12 production serving boundary.

The serving API is intentionally scoped to Saudi Arabia (`SA`) and exposes search, grounded answers, structured extraction, canonical document reads, readiness, and model capability metadata. Missing local corpus/model assets produce degraded readiness instead of opaque startup failures.

## Quick start

```bash
uv sync --locked --dev
uv run kawaneen --version
uv run kawaneen doctor
uv run kawaneen api serve
make check
```

See [the API guide](docs/api.md), [development.md](docs/development.md), and [the Phase 12 report](docs/reports/phase-12-api-report.md) for the serving contract and verification record.

## Product Interface

Phase 13 adds a recruiter-facing Streamlit workspace over the Phase 12 HTTP API. It is an evidence-first legal research interface—not a chatbot—with four screens:

- Search: ranked Saudi evidence, literal query highlighting, metadata, and returned-evidence refinement.
- Ask: grounded answers with a prominent citation rail, exact quotes, canonical-unit inspection, and intentional abstention states.
- Extract: paste, upload, or paginated corpus sources with bounded segmentation, visible hybrid limitations, and JSON/CSV exports.
- Evaluation: live readiness, provenance-hashed tracked metrics, and current-session latency labelled as non-benchmark.

Run the live API and UI as two processes:

```bash
make api-serve
make ui-serve
```

For a public, synthetic portfolio walkthrough use:

```bash
make ui-demo
```

Demo mode is labelled persistently as `DEMO DATA`; it never masquerades as a live system. Fixtures and visual artifacts are synthetic only. The UI uses the Phase 12 `/v1` HTTP boundary and does not import model-serving or retrieval runtime services.
