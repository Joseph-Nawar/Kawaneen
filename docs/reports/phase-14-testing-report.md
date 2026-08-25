# Phase 14 testing hardening report

This report records the public synthetic testing hardening on branch
`phase14/testing-hardening`, based on main
`294c8c171abc0e93d826ffbb6d2cf019c63e6d44`.

## Scope and audit

The authoritative main collected 877 tests. Existing Phase 7–13 tests already
covered most normalization, article parsing, chunking, RRF, NumPy exact index,
serving, grounding, citation, schema, and abstention contracts. Phase 14 adds
only the material gaps: explicit markers/commands, a text-based synthetic PDF,
layered deterministic integration, 20 public regression cases, a configuration
lock, cache-only model verification, a test-only Compose harness, and testing
documentation.

## Artifacts

- Unit hardening: `tests/test_phase14_unit_contracts.py` (19 tests).
- Integration: `tests/integration/` (PDF/chunks, NumPy index, retrieval/answer,
  and citation tests).
- Regression: `data/regression/phase14_cases.json` (20 public synthetic cases).
- Lock: `data/manifests/testing/phase14_regression_lock.json`.
- Fixture: `tests/fixtures/phase14/synthetic_appeals_regulation.pdf` plus the
  deliberately corrupt PDF failure fixture.
- Docker test harness: `docker-compose.e2e.yml` and `docker/e2e/`.

## Integrity constraints

No Qdrant, HOLDOUT access, evaluation rerun, tuning, model download, private
corpus, secret, or machine-local path was added. The frozen Phase 7–13 result
artifacts were not modified. The fixture and regression corpus are fictional
synthetic text only.

## Final verification record

The authoritative base SHA is
`294c8c171abc0e93d826ffbb6d2cf019c63e6d44`. The implementation head verified
by the fresh remote run is
`f2745950438b5f91bcbcf01b52d403c772ac4834`.

Local verification on macOS/Python 3.12:

- `make check`: 881 passed, 1 skipped, 42 deselected; 85.11% branch coverage.
- `make test-unit`: 860 passed, 1 skipped.
- `make test-integration`: 16 passed.
- `make test-regression`: 5 passed, covering all 20 public cases and the
  configuration lock.
- `make test-model-regression`: 1 passed from the existing local BGE-M3 and
  BGE reranker cache; no download occurred.
- `tests/test_public_ci_policy.py`: 1 passed.

PR and remote CI:

- PR [#10](https://github.com/Joseph-Nawar/Kawaneen/pull/10) is open against
  `main`, not merged, with merge state `CLEAN`/`MERGEABLE` at verification.
- Workflow run
  [32891307151](https://github.com/Joseph-Nawar/Kawaneen/actions/runs/32891307151)
  passed all three jobs: quality Python 3.11 (1m40s), quality Python 3.12
  (1m56s), and `e2e-compose` (31s).
- The successful Compose job exercised health, search, grounded answer,
  verified citation/display, and deliberate abstention in the public
  synthetic stack.

Docker details:

- The local Docker daemon reported `linux/arm64` (Docker 29.7.2), and the
  Compose definition contains no `platform: linux/amd64` override.
- The exact local Compose command was attempted after implementation and
  again after the CI fix. Both local attempts were blocked before image build
  by Docker Hub registry-auth `DeadlineExceeded` timeouts; the second attempt
  also confirmed the CI-discovered dependency fix in the tree. The cleanup
  command `docker compose -f docker-compose.e2e.yml down -v --remove-orphans`
  was run, leaving no project containers or volumes.

Integrity checks passed: no Qdrant, HOLDOUT access, tuning, model download,
private corpus, secret, or machine-local path was added; frozen Phase 7–13
result artifacts were not modified; and `git diff --check` is clean. The
report update itself is documentation-only after the tested implementation
head above.
