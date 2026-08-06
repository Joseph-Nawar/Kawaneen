# AGENTS.md

## Project boundary

Kawaneen Phase 0 is foundation only. Do not add document parsing, NLP, retrieval, RAG, APIs, Streamlit, Docker, models, vector databases, secrets, datasets, or network-dependent tests without an explicitly approved phase change.

## Development

Use Python 3.12 locally and support Python `>=3.11,<3.13`. Use `uv` for synchronization and commands. Run `make check` before handoff. Keep imports and module loading free of filesystem and network side effects.

## Data and secrets

Never commit `.env`, credentials, raw source material, external downloads, interim data, or processed data. Keep only version-controlled metadata under `data/manifests` and `data/evaluation`.
