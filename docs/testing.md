# Testing guide

Phase 14 separates fast contracts from subsystem integration, deterministic
behavior locks, optional model-cache checks, and the Docker Compose test
harness. Public tests are synthetic and hermetic; they do not read private
corpora, HOLDOUT cases, secrets, or download models.

## Commands

- `make test-unit` runs unmarked public unit contracts.
- `make test-integration` runs deterministic PDF, indexing, retrieval, answer,
  grounding, and citation integration tests.
- `make test-regression` runs the 20 public synthetic behavioral cases and
  verifies the read-only configuration lock.
- `make test-model-regression` runs BGE-M3 and BGE reranker checks only when
  both frozen revisions are already in the local Hugging Face model cache. It
  never downloads; an unavailable cache is an explicit skip.
- `make test-e2e` builds and runs the public Docker Compose harness, then
  removes its containers, networks, volumes, and orphans.
- `make test-e2e-private` is an optional `private_artifact` smoke tier for
  local Phase 12 assets, frozen model caches, and Ollama.
- `make check` remains the fast developer gate: format, lint, Pyright, source
  validation, and the public 85% branch-coverage suite. Compose E2E is not in
  this target.

## Layers

Unit tests protect pure contracts such as Arabic normalization policy behavior,
article labels, source spans, weighted RRF, schema rejection, citation data,
and pre-generation abstention. Integration tests wire the real parsing,
normalization, chunking, `NumpyExactIndex`, BM25/RRF serving, grounding,
answerability, citation, API, and UI presentation primitives with deterministic
adapters. Regression tests freeze user-visible outcomes rather than model
floating-point scores. E2E tests start the same API contract in a native
multi-architecture Compose harness.

The public E2E image uses an official Python 3.12 slim base and no
`platform: linux/amd64`, Rosetta, GPU, Qdrant, Redis, Postgres, Ollama, cloud
service, or private artifact. This supports Apple Silicon/M5 `linux/arm64` and
GitHub Ubuntu `linux/amd64` through the same Compose file. The public index is
`NumpyExactIndex`; Qdrant is intentionally not introduced because it is not
part of Kawaneen's shipped retrieval architecture.

## Updating a regression baseline intentionally

Run `make test-regression`, review the full behavioral differences, update the
synthetic case or lock explicitly, and document why in
`docs/reports/phase-14-testing-report.md`. The lock records source/configuration
hashes for normalization, parsing, chunking, dense/reranker identities and
revisions, fusion, generation, answerability, and fixture/case hashes. Tests do
not rewrite snapshots or accept failures automatically.

## Test matrix

| Requirement | Layer | Public CI | Model cache | Private data |
| --- | --- | --- | --- | --- |
| normalization, article, schema, fusion, abstention contracts | unit | yes | no | no |
| PDF to canonical units and chunks | integration | yes | no | no |
| chunks to `NumpyExactIndex` and query retrieval | integration | yes | no | no |
| grounded answer and verified citation | integration | yes | no | no |
| 20 synthetic observable behavior cases | regression | yes | no | no |
| BGE-M3 and BGE reranker frozen behavior | model_artifact | no | yes | no |
| Compose health/search/answer/citation/display/abstention | e2e | separate Ubuntu job | no | no |
| Phase 12 real-stack smoke | e2e + private_artifact | no | yes | yes |

## Compose usage

The stack is test-only:

```bash
make test-e2e
```

The underlying command is:

```bash
docker compose -f docker-compose.e2e.yml up --build --abort-on-container-exit --exit-code-from e2e
```

Cleanup is always attempted with:

```bash
docker compose -f docker-compose.e2e.yml down -v --remove-orphans
```
