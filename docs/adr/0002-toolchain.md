# ADR 0002: Python and Toolchain

## Decision

Use Python 3.12 locally, support `>=3.11,<3.13`, manage dependencies with uv, build with Hatchling, lint and format with Ruff, type-check with strict Pyright, and test with pytest plus branch-aware coverage.

## Context

The foundation needs a small, reproducible toolchain with fast local checks and a lockfile suitable for CI.

## Consequences

Runtime dependencies stay limited to Pydantic, pydantic-settings, and structlog. Development tools are isolated in the `dev` dependency group. CI uses locked synchronization.
