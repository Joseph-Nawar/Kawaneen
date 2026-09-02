.PHONY: help install install-observability format lint typecheck test test-unit test-integration test-regression test-model-regression test-e2e test-e2e-private test-public test-private check doctor api-serve api-serve-observed mlflow-serve ui-serve ui-demo phase16-identity phase16-reproduce phase16-verify sources-validate sources-summary data-plan data-acquire-alarb data-import-arabiccr data-verify data-audit data-audit-statutory data-manifest data-status data-rebuild-auto data-rebuild corpus-plan corpus-build corpus-validate corpus-inventory corpus-statutory-status corpus-duplicate-diagnostics corpus-gaps parsing-preflight parsing-benchmark parsing-diagnose normalization-plan normalization-run normalization-validate normalization-sensitivity chunking-plan chunking-build chunking-experiment chunking-validate evaluation-plan evaluation-build-draft evaluation-build-draft-v3 evaluation-build-draft-v4 evaluation-build-draft-v5 evaluation-build-final-candidate evaluation-export-review evaluation-import-review evaluation-validate evaluation-freeze evaluation-freeze-ai-reviewed evaluation-stats extraction-status extraction-prepare extraction-validate extraction-deterministic phase15-plan phase15-freeze phase15-synthesize phase15-embedding phase15-dialect phase15-reranking phase15-generation-preflight phase15-generation phase15-counterfactuals phase15-latency phase15-review-prepare phase15-review phase15-review-status phase15-finalize clean

help:
	@printf '%s\n' 'Kawaneen development commands:'
	@printf '%s\n' '  make install   uv sync --locked --dev --extra ui'
	@printf '%s\n' '  make install-observability  uv sync --locked --dev --extra ui --group observability'
	@printf '%s\n' '  make format    uv run ruff format .'
	@printf '%s\n' '  make lint      uv run ruff check .'
	@printf '%s\n' '  make typecheck uv run pyright'
	@printf '%s\n' '  make test      uv run pytest'
	@printf '%s\n' '  make test-unit  fast public unit tests only'
	@printf '%s\n' '  make test-integration  public deterministic integration tests'
	@printf '%s\n' '  make test-regression  public hermetic regression suite'
	@printf '%s\n' '  make test-model-regression  cache-only frozen model regression'
	@printf '%s\n' '  make test-e2e  Docker Compose public deterministic E2E'
	@printf '%s\n' '  make test-e2e-private  optional local private Phase 12 smoke'
	@printf '%s\n' '  make test-public  public hermetic tests with the 85% coverage gate'
	@printf '%s\n' '  make test-private  local private-artifact integration tests (set private roots as needed)'
	@printf '%s\n' '  make check     format check, lint, typecheck, tests'
	@printf '%s\n' '  make doctor    uv run kawaneen doctor'
	@printf '%s\n' '  make api-serve uv run kawaneen api serve'
	@printf '%s\n' '  make api-serve-observed  KAWANEEN_OBSERVABILITY_ENABLED=true KAWANEEN_MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run kawaneen api serve'
	@printf '%s\n' '  make mlflow-serve  start local loopback MLflow with SQLite storage'
	@printf '%s\n' '  make phase16-identity  verify the tracked serving identity'
	@printf '%s\n' '  make phase16-reproduce  reconstruct the public result table'
	@printf '%s\n' '  make phase16-verify  verify identity and public result reproduction'
	@printf '%s\n' '  make ui-serve   uv run streamlit run src/kawaneen/ui/app.py'
	@printf '%s\n' '  make ui-demo    KAWANEEN_UI_MODE=demo uv run streamlit run src/kawaneen/ui/app.py'
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
	@printf '%s\n' '  make corpus-plan       uv run kawaneen corpus plan'
	@printf '%s\n' '  make corpus-build      uv run kawaneen corpus build (local data; not CI)'
	@printf '%s\n' '  make corpus-validate   uv run kawaneen corpus validate'
	@printf '%s\n' '  make corpus-inventory  uv run kawaneen corpus inventory'
	@printf '%s\n' '  make corpus-statutory-status  uv run kawaneen corpus statutory-status'
	@printf '%s\n' '  make corpus-duplicate-diagnostics  uv run kawaneen corpus duplicate-diagnostics'
	@printf '%s\n' '  make corpus-gaps       uv run kawaneen corpus gaps'
	@printf '%s\n' '  make parsing-benchmark uv run kawaneen parsing benchmark (private pages required)'
	@printf '%s\n' '  make parsing-preflight uv run kawaneen parsing preflight'
	@printf '%s\n' '  make parsing-diagnose FILE=...  uv run kawaneen parsing diagnose --path FILE (optional dependencies)'
	@printf '%s\n' '  make normalization-plan   uv run kawaneen normalization plan'
	@printf '%s\n' '  make normalization-run    uv run kawaneen normalization run (private local corpus)'
	@printf '%s\n' '  make normalization-validate  uv run kawaneen normalization validate'
	@printf '%s\n' '  make normalization-sensitivity  uv run kawaneen normalization sensitivity'
	@printf '%s\n' '  make chunking-plan             uv run kawaneen chunking plan'
	@printf '%s\n' '  make chunking-build            uv run kawaneen chunking build (private local corpus)'
	@printf '%s\n' '  make chunking-experiment       uv run kawaneen chunking experiment (private local corpus)'
	@printf '%s\n' '  make chunking-validate         uv run kawaneen chunking validate'
	@printf '%s\n' '  make evaluation-plan           uv run kawaneen evaluation plan'
	@printf '%s\n' '  make evaluation-build-draft    uv run kawaneen evaluation build-draft (private)'
	@printf '%s\n' '  make evaluation-build-draft-v3 FILE=...  uv run kawaneen evaluation build-draft-v3 --review-file FILE (private)'
	@printf '%s\n' '  make evaluation-build-draft-v4 FILE=...  uv run kawaneen evaluation build-draft-v4 --review-file FILE (private)'
	@printf '%s\n' '  make evaluation-build-draft-v5 FILE=...  uv run kawaneen evaluation build-draft-v5 --review-file FILE (private)'
	@printf '%s\n' '  make evaluation-build-final-candidate FILE=...  uv run kawaneen evaluation build-final-candidate --patch-file FILE (private)'
	@printf '%s\n' '  make evaluation-export-review  uv run kawaneen evaluation export-review (private)'
	@printf '%s\n' '  make evaluation-import-review  uv run kawaneen evaluation import-review FILE=... (private)'
	@printf '%s\n' '  make evaluation-validate       uv run kawaneen evaluation validate'
	@printf '%s\n' '  make evaluation-freeze         uv run kawaneen evaluation freeze (review-gated)'
	@printf '%s\n' '  make evaluation-freeze-ai-reviewed  uv run kawaneen evaluation freeze-ai-reviewed'
	@printf '%s\n' '  make evaluation-stats          uv run kawaneen evaluation stats'
	@printf '%s\n' '  make extraction-status         uv run kawaneen extraction status'
	@printf '%s\n' '  make extraction-prepare        uv run kawaneen extraction prepare-annotations'
	@printf '%s\n' '  make extraction-validate       uv run kawaneen extraction validate-annotations --split dev'
	@printf '%s\n' '  make extraction-deterministic  uv run kawaneen extraction run-deterministic --split dev'
	@printf '%s\n' '  make phase15-freeze            uv run kawaneen phase15 freeze'
	@printf '%s\n' '  make phase15-review-prepare    uv run kawaneen phase15 review-prepare'
	@printf '%s\n' '  make phase15-review            uv run streamlit run src/kawaneen/phase15/review_app.py'
	@printf '%s\n' '  make phase15-review-status     uv run kawaneen phase15 review-status'
	@printf '%s\n' '  make phase15-finalize           uv run kawaneen phase15 finalize'
	@printf '%s\n' '  make clean     remove safe local build and test artifacts'

install:
	uv sync --locked --dev --extra ui

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run pyright

test:
	uv run pytest

test-unit:
	uv run pytest -m "not integration and not regression and not model_artifact and not e2e and not private_artifact" --no-cov

test-integration:
	uv run pytest -m integration --no-cov

test-regression:
	uv run pytest -m regression --no-cov

test-model-regression:
	uv run pytest -m model_artifact --no-cov

test-e2e:
	trap 'docker compose -f docker-compose.e2e.yml down -v --remove-orphans' EXIT; docker compose -f docker-compose.e2e.yml up --build --abort-on-container-exit --exit-code-from e2e

test-e2e-private:
	uv run pytest -m "e2e and private_artifact" --no-cov

test-public:
	uv run pytest -m "not private_artifact and not model_artifact and not e2e" --cov=kawaneen --cov-branch --cov-report=term-missing --cov-fail-under=85

test-private:
	uv run pytest -m private_artifact --no-cov

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright
	uv run kawaneen sources validate
	$(MAKE) test-public

doctor:
	uv run kawaneen doctor

api-serve:
	uv run kawaneen api serve

install-observability:
	uv sync --locked --dev --extra ui --group observability

mlflow-serve:
	mkdir -p artifacts/observability/mlartifacts
	uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///artifacts/observability/mlflow.db --default-artifact-root artifacts/observability/mlartifacts

api-serve-observed:
	KAWANEEN_OBSERVABILITY_ENABLED=true KAWANEEN_MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run kawaneen api serve

phase16-identity:
	uv run kawaneen phase16 identity

phase16-reproduce:
	uv run kawaneen phase16 reproduce

phase16-verify:
	uv run kawaneen phase16 verify

ui-serve:
	uv run streamlit run src/kawaneen/ui/app.py

ui-demo:
	KAWANEEN_UI_MODE=demo uv run streamlit run src/kawaneen/ui/app.py

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

corpus-plan:
	uv run kawaneen corpus plan

corpus-build:
	uv run kawaneen corpus build

corpus-validate:
	uv run kawaneen corpus validate

corpus-inventory:
	uv run kawaneen corpus inventory

corpus-statutory-status:
	uv run kawaneen corpus statutory-status

corpus-duplicate-diagnostics:
	uv run kawaneen corpus duplicate-diagnostics

corpus-gaps:
	uv run kawaneen corpus gaps

parsing-benchmark:
	uv run kawaneen parsing benchmark

parsing-preflight:
	uv run kawaneen parsing preflight

parsing-diagnose:
	uv run kawaneen parsing diagnose --path "$${FILE:?set FILE to a one-page PDF path}"

normalization-plan:
	uv run kawaneen normalization plan

normalization-run:
	uv run kawaneen normalization run

normalization-validate:
	uv run kawaneen normalization validate

normalization-sensitivity:
	uv run kawaneen normalization sensitivity

chunking-plan:
	uv run kawaneen chunking plan

chunking-build:
	uv run kawaneen chunking build

chunking-experiment:
	uv run kawaneen chunking experiment

chunking-validate:
	uv run kawaneen chunking validate

evaluation-plan:
	uv run kawaneen evaluation plan

evaluation-build-draft:
	uv run kawaneen evaluation build-draft

evaluation-build-draft-v3:
	uv run kawaneen evaluation build-draft-v3 --review-file "$${FILE:?set FILE to the external source-review JSONL path}"

evaluation-build-draft-v4:
	uv run kawaneen evaluation build-draft-v4 --review-file "$${FILE:?set FILE to the external v3 adjudication JSONL path}"

evaluation-build-draft-v5:
	uv run kawaneen evaluation build-draft-v5 --review-file "$${FILE:?set FILE to the final v4 adjudication JSONL path}"

evaluation-build-final-candidate:
	uv run kawaneen evaluation build-final-candidate --patch-file "$${FILE:?set FILE to the final literal patch JSONL path}"

evaluation-export-review:
	uv run kawaneen evaluation export-review

evaluation-import-review:
	uv run kawaneen evaluation import-review --file "$${FILE:?set FILE to a review packet path}"

evaluation-validate:
	uv run kawaneen evaluation validate

evaluation-freeze:
	uv run kawaneen evaluation freeze

evaluation-freeze-ai-reviewed:
	uv run kawaneen evaluation freeze-ai-reviewed

evaluation-stats:
	uv run kawaneen evaluation stats

extraction-status:
	uv run kawaneen extraction status

extraction-prepare:
	uv run kawaneen extraction prepare-annotations

extraction-validate:
	uv run kawaneen extraction validate-annotations --split dev

extraction-deterministic:
	uv run kawaneen extraction run-deterministic --split dev

phase15-plan:
	uv run kawaneen phase15 plan

phase15-freeze:
	uv run kawaneen phase15 freeze

phase15-synthesize:
	uv run kawaneen phase15 synthesize

phase15-embedding:
	uv run kawaneen phase15 embedding

phase15-dialect:
	uv run kawaneen phase15 dialect-prepare
	uv run kawaneen phase15 dialect-evaluate

phase15-reranking:
	uv run kawaneen phase15 reranking

phase15-generation-preflight:
	uv run kawaneen phase15 generation-preflight

phase15-generation:
	uv run kawaneen phase15 generation-run

phase15-counterfactuals:
	uv run kawaneen phase15 counterfactuals

phase15-latency:
	uv run kawaneen phase15 latency

phase15-review-prepare:
	uv run kawaneen phase15 review-prepare

phase15-review:
	uv run streamlit run src/kawaneen/phase15/review_app.py

phase15-review-status:
	uv run kawaneen phase15 review-status

phase15-finalize:
	uv run kawaneen phase15 finalize

clean:
	find . -maxdepth 1 -type d \( -name .pytest_cache -o -name .ruff_cache -o -name htmlcov -o -name dist -o -name build \) -exec rm -rf {} +
	find . -maxdepth 1 -type f \( -name .coverage -o -name coverage.xml \) -exec rm -f {} +
