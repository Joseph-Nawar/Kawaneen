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

Integration, regression, and public E2E share `tests/phase14_support.py`:
PDF parser-health probe, canonical units, normalization, legal chunking, BM25
plus deterministic dense retrieval, weighted RRF, deterministic reranking,
serving depth, grounding, answerability, generation, and citation verification.

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
`b2757be`. The report update after that verification is documentation-only.

Local verification on macOS/Python 3.12:

- `make check`: 883 passed, 1 skipped, 42 deselected; 85.14% branch coverage.
- `make test-unit`: 860 passed, 1 skipped.
- `make test-integration`: 17 passed.
- `make test-regression`: 6 passed, covering all 20 public cases through the
  shared hybrid stack and the configuration lock.
- `make test-model-regression`: 1 passed from the existing local BGE-M3 and
  BGE reranker cache across all 20 cases; no model download occurred.
- `tests/test_public_ci_policy.py`: 1 passed.

PR and remote CI:

- PR [#10](https://github.com/Joseph-Nawar/Kawaneen/pull/10) is open against
  `main`, not merged, with merge state `CLEAN`/`MERGEABLE` at verification.
- Workflow run
  [32896051326](https://github.com/Joseph-Nawar/Kawaneen/actions/runs/32896051326)
  passed all three jobs: quality Python 3.11 (1m54s), quality Python 3.12
  (2m08s), and `e2e-compose` (37s).
- The successful Compose job exercised health, search, grounded answer,
  verified citation/display, and deliberate abstention in the public
  synthetic stack.

Docker details:

- The local Docker daemon reported `linux/aarch64`; the built image reported
  `arm64/linux` (Docker 29.7.2), and the Compose definition contains no
  `platform: linux/amd64` override.
- After a bounded retry, `docker pull python:3.12-slim` succeeded and the exact
  local `make test-e2e` command passed on the ARM64 image. It exercised health,
  search, grounded Article 14 answer, verified citation/display, and deliberate
  abstention. The Compose cleanup trap left no project containers or volumes.

Integrity checks passed: no Qdrant, HOLDOUT access, tuning, model download,
private corpus, secret, or machine-local path was added; frozen Phase 7–13
result artifacts were not modified; and `git diff --check` is clean. The
report update itself is documentation-only after the tested implementation
head above.
