# Architecture

Phase 0 has three small runtime boundaries:

1. `kawaneen.cli` owns the dependency-free `argparse` command surface.
2. `kawaneen.core.config` owns validated Pydantic Settings with a `KAWANEEN_` environment prefix; constructing settings does not create directories.
3. `kawaneen.core.logging` owns idempotent structlog setup through standard logging, with console and JSON renderers.

The package uses a `src` layout and Hatchling builds it as an installable distribution. Tests exercise public behavior through imports, subprocess CLI calls, settings construction, and captured logs.

Later phases may add domain modules behind these boundaries. Planned legal ingestion, parsing, NLP, retrieval, RAG, interfaces, and model infrastructure are deliberately absent from this repository state.
