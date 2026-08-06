# Phase 0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the installable, tested Kawaneen Python foundation with configuration, logging, CLI diagnostics, project automation, safeguards, CI, and accurate Phase 0 documentation.

**Architecture:** Keep a small `src/kawaneen` package with a dependency-free CLI shell, Pydantic Settings configuration, and structlog-backed logging. Keep future legal intelligence capabilities explicitly out of scope and represent only their planned boundaries in documentation.

**Tech Stack:** Python 3.12 locally; Python `>=3.11,<3.13` supported; uv; Hatchling; Pydantic; pydantic-settings; structlog; pytest; coverage; Ruff; Pyright; pre-commit; GitHub Actions.

## Global Constraints

- Display, repository, distribution, and import name: `kawaneen`.
- Use an installable `src` layout at `src/kawaneen`.
- Runtime dependencies are only Pydantic, pydantic-settings, and structlog.
- Do not implement parsing, NLP, retrieval, RAG, APIs, Streamlit, Docker, models, or vector databases.
- Do not add paid services, secrets, datasets, models, or network-dependent tests.
- Use strict Pyright for the package and curated Ruff rules, never `ALL`.
- Require minimum branch-aware coverage of 85%.
- Loading settings and importing the package must not create directories or expose secrets.

---

### Task 1: Package behavior and tests

**Files:** Create `tests/test_package.py`, `tests/test_cli.py`, `tests/test_config.py`, `tests/test_logging.py`, and `tests/conftest.py`.

- [ ] Write tests first for version exposure, `--version`, `doctor`, settings defaults and `KAWANEEN_` overrides, invalid values, import-time filesystem silence, console logs, JSON logs, and repeated logging setup.
- [ ] Run the focused tests and confirm they fail because the package does not yet exist.
- [ ] Keep subprocess tests isolated from the repository environment and use temporary paths only in test setup.

### Task 2: Core implementation

**Files:** Create `src/kawaneen/__init__.py`, `src/kawaneen/__main__.py`, `src/kawaneen/cli.py`, `src/kawaneen/core/__init__.py`, `src/kawaneen/core/config.py`, and `src/kawaneen/core/logging.py`.

- [ ] Implement a dependency-free argparse CLI with `--version` and `doctor`.
- [ ] Implement validated Pydantic Settings fields for environment, log level, log format, data directory, and artifacts directory with `KAWANEEN_` prefix and no filesystem work during load.
- [ ] Implement idempotent structlog configuration integrated with standard logging, with console and JSON renderers.
- [ ] Run focused tests and then the complete test suite.

### Task 3: Packaging and developer tooling

**Files:** Create `pyproject.toml`, `.python-version`, `.env.example`, `.gitignore`, `LICENSE`, `AGENTS.md`, `Makefile`, `.pre-commit-config.yaml`, and `.github/workflows/ci.yml`.

- [ ] Configure Hatchling, runtime/development dependency groups, the console entry point, Ruff, pytest/coverage, and strict Pyright.
- [ ] Add documented Make targets and safe data-cleaning behavior.
- [ ] Add fast pre-commit hygiene, large-file/private-key protection, Ruff lint, and Ruff format hooks.
- [ ] Add locked uv synchronization and CI checks for pushes to `main`, pull requests, and manual runs.

### Task 4: Documentation and data safeguards

**Files:** Create `README.md`, `docs/project-charter.md`, `docs/architecture.md`, `docs/development.md`, `docs/phases/phase-00-foundation.md`, `docs/adr/0001-project-identity-and-scope.md`, `docs/adr/0002-toolchain.md`, `docs/adr/0003-src-layout-and-framework-light-architecture.md`, and `docs/reports/phase-00-foundation-report.md`; create `data/manifests/.gitkeep` and `data/evaluation/.gitkeep`.

- [ ] Document the current foundation, planned later capabilities, boundaries, commands, and manual prerequisites without claiming legal search or RAG exists.
- [ ] Ignore raw, external, interim, and processed data while preserving version-controlled manifests and evaluation metadata.

### Task 5: Verification and handoff

- [ ] Generate `uv.lock` with the requested locked development sync when uv is available.
- [ ] Run every requested command, record actual outputs and any environment-based deviation in the Phase 0 report, and fix failures.
- [ ] Run `git diff --check`, inspect the final file map and Git status, and commit only if Git identity is already configured.
