# Kawaneen

Kawaneen is the foundation for a future jurisdiction-aware Arabic legal and regulatory intelligence system. Phase 0 provides an installable Python package, validated settings, structured logging, a small diagnostic CLI, and developer quality gates.

Legal document parsing, NLP, search, retrieval, RAG, APIs, and model-backed functionality are planned for later phases and are not implemented here.

## Quick start

```bash
uv sync --locked --dev
uv run kawaneen --version
uv run kawaneen doctor
make check
```

See [development.md](docs/development.md) for the complete workflow and [the Phase 0 report](docs/reports/phase-00-foundation-report.md) for verification results.
