# Phase 16: Observability and Reproducibility

## Goal

Phase 16 adds a small, opt-in observability layer around the existing serving
pipeline and a public reconstruction contract for six reported aggregate
results. It observes Phase 15/earlier behavior; it does not change retrieval,
reranking, prompts, generation, answerability, citation verification, model
selection, or API response contracts.

## Architecture

`ServingIdentity` is derived from the tracked Phase 7/8/10 serving locks and
current prompt/policy constants. Its `configuration_version` is the SHA-256
of canonical UTF-8 JSON with sorted keys and compact separators, excluding the
version field itself. It contains no paths, timestamps, hostnames, or Git
state. The tracked identity is at
`data/manifests/observability/phase16_serving_identity.json`.

The existing middleware request ID remains the correlation ID. The API opens
the root trace inside the synchronous `asyncio.to_thread` worker, so nested
retrieval and answer spans share one MLflow trace context:

```text
kawaneen.search
  retrieval.first_stage
  retrieval.rerank

kawaneen.answer
  retrieval.first_stage
  retrieval.rerank
  context.assemble
  answerability.policy
  generation.stage_d
  citation.verify

kawaneen.extract
  extraction
```

`/health`, `/models`, and `/documents` are not traced. Stage latency is the
MLflow span duration; it is environment-dependent and is not a golden
deterministic result. The client-visible `latency_ms` remains unchanged.

## What is tracked

Search and answer roots carry the trace schema, existing request ID,
configuration/corpus versions, embedding model/revision, retrieval strategy,
reranker model/revision, and sanitized request metadata such as character
count, jurisdiction, and requested limit. Answer roots additionally carry
generator provider/model/revision, prompt template/version hash, and
answerability policy version.

Retrieval spans carry first-stage parameters, ordered fused chunk IDs, every
reranker score with prior fused rank, final returned IDs, and requested limit.
Answer spans carry context counts, the authoritative answerability decision,
generation status, and citation-verification counts/status. Extraction carries
mode, capability status, and available provider/model metadata.

No raw user query, query hash, retrieved legal text, generated answer, quoted
citation text, extraction input, or extraction output is sent to MLflow. The
privacy regression test serializes all observer-created attributes, inputs, and
outputs and checks for sentinel raw content.

## Authority and local MLflow

Tracked repository configuration and aggregate artifacts are the source of
reproducibility truth. MLflow is a local inspection/tracing interface only. It
is an optional `observability` dependency group and is lazy-imported, so normal
Kawaneen imports, tests, and serving do not require MLflow. When explicitly
enabled, startup verifies the configured tracking server; a later telemetry
write failure is logged as a warning and cannot replace a domain result or
exception.

Start local MLflow with SQLite metadata and local artifacts:

```bash
make install-observability
make mlflow-serve
```

Storage is under ignored `artifacts/observability/` and no MLflow database or
runtime artifact is committed.

## Public reproduction workflow

```bash
uv sync --locked --dev
make phase16-identity
make phase16-reproduce
make phase16-verify
```

`phase16-reproduce` loads the declarative
`data/manifests/observability/phase16_reproduction_config.json`, verifies every
source SHA-256, traverses arrays of object keys, and writes the ignored local
copy `artifacts/observability/reproduced_results.csv`. It compares those bytes
with the tracked `data/evaluation/phase16_reported_results.csv`.

The optional supplemental run is:

```bash
kawaneen phase16 reproduce --mlflow
```

It logs the serving configuration version, reproduction-config/table hashes,
source count, Git/Python/MLflow versions, six stable metrics, and only the
public reproduction config and CSV.

## Six reproduced results

| Result | Value | Source |
| --- | ---: | --- |
| `phase5_fixed256_ndcg10` | 0.6394436819923248 | `phase5_chunking_metrics.json` |
| `phase5_structure_ndcg10` | 0.683666296657345 | `phase5_chunking_metrics.json` |
| `phase15_arabic_vs_bge_recall10_delta` | 0.04609929078014184 | `phase15_embedding_metrics.json` |
| `phase15_reranker_recall10_delta` | 0.043478260869565216 | `phase15_reranking_metrics.json` |
| `phase15_citation_defect_reduction` | 0.725 | `phase15_citation_counterfactual.json` |
| `phase15_fallback_invalid_generation_rate` | 1.0 | `phase15_generator_metrics.json` |

The deterministic table SHA-256 for this release is
`eb86dcd4896c3c5ceb12e0b5b236760cddd2910d0ad6e847fdb6c1dd7d71c78d`.

## Limitations and governance

Public repository users can deterministically reconstruct the reported
aggregate result table from tracked source artifacts and verify source hashes.
Full raw-data experiment reruns require the corresponding private/local
evaluation assets. Latency is traced for operational inspection but is not a
reproducible reported result. Phase 16 does not rerun Phase 15 experiments,
access protected HOLDOUT data, or promote a model.

## Artifact map

- `src/kawaneen/observability/identity.py`: immutable serving identity and hash verification.
- `src/kawaneen/observability/tracing.py`: no-op/manual MLflow observer boundary.
- `src/kawaneen/observability/reproducibility.py`: source validation, CSV generation, and optional MLflow reproduction run.
- `data/manifests/observability/phase16_serving_identity.json`: tracked serving identity.
- `data/manifests/observability/phase16_reproduction_config.json`: six declarative result definitions and source hashes.
- `data/evaluation/phase16_reported_results.csv`: tracked deterministic result table.
- `artifacts/observability/`: ignored local MLflow database, artifacts, and reproduced CSV.

## Verification record

The implementation includes focused identity, tracing/privacy, branch-status,
API, and reproduction tests. The recorded local smoke used MLflow 3.15.2 with
a synthetic request: the client observed a trace with request ID
`smoke-req`, configuration identity `3fcd52f794ca6402c5d1d6f8be3a2aa9487ea79ac8ccd967392be3304638c83c`,
and nested `retrieval.first_stage`. The supplemental reproduction run created
the `kawaneen-reproducibility` experiment with six metrics and the two public
artifacts. Public CI must not depend on a running MLflow server.
