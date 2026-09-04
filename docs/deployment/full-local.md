# Full local deployment

This is the authoritative frozen Kawaneen serving profile. It runs the existing
FastAPI and Streamlit boundary with BGE-M3, the Phase 8 frozen hybrid pipeline,
exact Qdrant dense search, the frozen BGE reranker, Stage-D Qwen3 4B through
Ollama, the existing verifier, and Phase 16 MLflow traces.

## Prerequisites

- Docker Desktop or compatible Docker Compose with native Apple Silicon support.
- Python/uv for local tooling; Python support remains `>=3.11,<3.13`.
- The local frozen private artifact root containing Phase 6/7/10/11 serving
  assets. Keep it outside Git and mount it read-only.
- Disk space for the frozen BGE/reranker assets, the Qwen Ollama model, Qdrant
  storage, and MLflow state. Measure this on the target machine; it is not a
  repository guarantee.

Set an external artifact root when needed:

```bash
export KAWANEEN_HOST_ARTIFACTS_DIR=/absolute/path/to/artifacts
docker compose up
```

The default is `./artifacts`. Host ports bind to loopback only: API `8000`,
UI `8501`, Qdrant `6333`, MLflow `5000`, and Ollama `11434`.

## Startup and verification

The canonical command is:

```bash
docker compose up
```

`qdrant-init` validates and idempotently seeds the Phase-17-owned collection
from `chunks.jsonl`, `vectors.npy`, and `ids.json`. `ollama-init` reads the
authoritative Phase 10 selection and pulls the frozen tag only when absent;
both jobs fail closed on identity mismatch. The API starts only after those
jobs and MLflow are healthy.

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/v1/health
curl -fsS http://127.0.0.1:8000/v1/models
```

See [tested API examples](api-examples.md) for request bodies. MLflow is
available at `http://127.0.0.1:5000`; a real `/v1/search` or `/v1/answer`
request creates the existing Phase 16 trace hierarchy without recording raw
queries, legal text, answers, or citation text.

## Teardown

```bash
docker compose down
```

Use `docker compose down -v` only when intentionally removing named Qdrant,
MLflow, and Ollama state. The Compose E2E file is a deterministic test harness
and is not the full deployment profile.

## Verification record

On 2026-09-04, the Apple-Silicon verification Docker engine reported
`12,526,370,816` bytes of memory (approximately 11.66 GiB). With the complete
private artifact root mounted read-only, the full frozen Compose stack was
successfully verified. `hf-model-init` completed the exact frozen BGE-M3 and
BGE reranker snapshots before API startup, including repair of an existing
partial reranker cache. This is a tested local observation, not a universal
hardware requirement.

The earlier 2026-09-03 run with approximately 7.75 GiB
(`8,319,504,384` bytes) OOM-killed `kawaneen-api` with exit 137 during startup
while loading the frozen retrieval models; the kernel reported approximately
5.7 GiB of API anonymous RSS. No precision, architecture, or frozen model
setting was changed to work around that limitation.

The initial 2026-09-03 configured-artifact attempt also lacked the required
private Phase 7 seed and Phase 10 Ollama lock, so its init jobs failed closed
before the API and UI could start. The Ollama container itself had the frozen
model tag and digest cached, and MLflow became healthy after pinning AnyIO to
the repository lock.
