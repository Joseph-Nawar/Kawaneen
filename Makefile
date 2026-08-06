.PHONY: help install format lint typecheck test check doctor sources-validate sources-summary clean

help:
	@printf '%s\n' 'Kawaneen development commands:'
	@printf '%s\n' '  make install   uv sync --locked --dev'
	@printf '%s\n' '  make format    uv run ruff format .'
	@printf '%s\n' '  make lint      uv run ruff check .'
	@printf '%s\n' '  make typecheck uv run pyright'
	@printf '%s\n' '  make test      uv run pytest'
	@printf '%s\n' '  make check     format check, lint, typecheck, tests'
	@printf '%s\n' '  make doctor    uv run kawaneen doctor'
	@printf '%s\n' '  make sources-validate  uv run kawaneen sources validate'
	@printf '%s\n' '  make sources-summary   uv run kawaneen sources summary'
	@printf '%s\n' '  make clean     remove safe local build and test artifacts'

install:
	uv sync --locked --dev

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run pyright

test:
	uv run pytest

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright
	uv run kawaneen sources validate
	uv run pytest

doctor:
	uv run kawaneen doctor

sources-validate:
	uv run kawaneen sources validate

sources-summary:
	uv run kawaneen sources summary

clean:
	find . -maxdepth 1 -type d \( -name .pytest_cache -o -name .ruff_cache -o -name htmlcov -o -name dist -o -name build \) -exec rm -rf {} +
	find . -maxdepth 1 -type f \( -name .coverage -o -name coverage.xml \) -exec rm -f {} +
