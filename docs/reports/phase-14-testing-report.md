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

The base SHA is `294c8c171abc0e93d826ffbb6d2cf019c63e6d44`. The final SHA, PR metadata, fresh CI status, coverage, Docker architecture,
Compose outcome, and cleanup result are recorded here after the final
verification commands and remote CI run. This report intentionally does not
invent unavailable remote or local Docker evidence.
