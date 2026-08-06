# Development

## Prerequisites

Install Python 3.12 and uv. The project supports Python `>=3.11,<3.13`; `.python-version` selects 3.12 for local uv environments.

## Common commands

```bash
uv sync --locked --dev
uv run kawaneen --version
uv run kawaneen doctor
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
uv run pre-commit run --all-files
make check
git diff --check
```

`make help` lists the same workflows. `make clean` removes only local build, coverage, and test-cache artifacts at the repository root; it does not remove source, documentation, Git metadata, or data metadata.

## Configuration

Settings default to development, INFO, console logs, `data`, and `artifacts`. Override with `KAWANEEN_ENVIRONMENT`, `KAWANEEN_LOG_LEVEL`, `KAWANEEN_LOG_FORMAT`, `KAWANEEN_DATA_DIRECTORY`, and `KAWANEEN_ARTIFACTS_DIRECTORY`. Settings loading validates values and does not create directories. Keep secrets out of environment examples and Git.

## Contribution gate

Add a focused test before behavior changes, keep branch-aware coverage at or above 85%, and run `make check` plus pre-commit before handoff.
