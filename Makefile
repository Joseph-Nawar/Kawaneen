.PHONY: help install format lint typecheck test check doctor sources-validate sources-summary data-plan data-acquire-alarb data-import-arabiccr data-verify data-audit data-audit-statutory data-manifest data-status data-rebuild-auto data-rebuild clean

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
	@printf '%s\n' '  make data-plan         uv run kawaneen data plan'
	@printf '%s\n' '  make data-acquire-alarb  uv run kawaneen data acquire alarb --purpose evaluation (network)'
	@printf '%s\n' '  make data-import-arabiccr FILE=...  uv run kawaneen data import-local arabiccr --file FILE --purpose local_research'
	@printf '%s\n' '  make data-verify       uv run kawaneen data verify'
	@printf '%s\n' '  make data-audit        uv run kawaneen data audit'
	@printf '%s\n' '  make data-audit-statutory  uv run kawaneen data audit-statutory saudi-moj-derived'
	@printf '%s\n' '  make data-manifest     uv run kawaneen data manifest build alarb'
	@printf '%s\n' '  make data-status       uv run kawaneen data status'
	@printf '%s\n' '  make data-rebuild-auto uv run kawaneen data rebuild --auto'
	@printf '%s\n' '  make data-rebuild      alias for data-rebuild-auto'
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

data-plan:
	uv run kawaneen data plan

data-acquire-alarb:
	uv run kawaneen data acquire alarb --purpose evaluation

data-import-arabiccr:
	uv run kawaneen data import-local arabiccr --file "$${FILE:?set FILE to a local ArabiCCR-dataset.csv path}" --purpose local_research

data-verify:
	uv run kawaneen data verify --source alarb

data-audit:
	uv run kawaneen data audit --source alarb

data-audit-statutory:
	uv run kawaneen data audit-statutory saudi-moj-derived

data-manifest:
	uv run kawaneen data manifest build alarb

data-status:
	uv run kawaneen data status

data-rebuild-auto:
	uv run kawaneen data rebuild --auto

data-rebuild: data-rebuild-auto

clean:
	find . -maxdepth 1 -type d \( -name .pytest_cache -o -name .ruff_cache -o -name htmlcov -o -name dist -o -name build \) -exec rm -rf {} +
	find . -maxdepth 1 -type f \( -name .coverage -o -name coverage.xml \) -exec rm -f {} +
