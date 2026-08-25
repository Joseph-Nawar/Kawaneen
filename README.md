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
