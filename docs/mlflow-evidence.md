# MLflow portfolio evidence

This is a safe, text-only export of existing verified Phase 16/17 local
observability and reproduction evidence. It is a portfolio artifact, not a
replacement for the local MLflow store.

## What is included

[`data/evaluation/portfolio_mlflow_evidence.json`](../data/evaluation/portfolio_mlflow_evidence.json)
contains:

- schema and export provenance;
- serving `configuration_version`, private-corpus version hash, model IDs and
  revisions, retrieval strategy, and prompt/answerability hashes;
- safe reproduction run/trace IDs already recorded by Phase 16;
- root and nested stage names with observed status; latency is marked as
  environment-dependent rather than invented;
- the six reproduced metric names/values, source-artifact count, and result
  table hash.

## What is explicitly excluded

The export contains no raw query text, query hash, legal evidence, generated
answer, extracted document text, quote text, private local path, credential,
stack trace, or MLflow database/artifact payload. The export does not add a
new tracking framework or rerun a research experiment.

## Reproducibility boundary

The tracked Phase 16 configuration and CSV remain the source of reproducibility
truth. Rebuild and verify them with:

```bash
make phase16-verify
```

The actual SQLite database and artifact store remain under ignored
`artifacts/observability/`. MLflow is optional and lazy-loaded; normal imports,
tests, and public CI do not require a running tracking server. Full raw-data
experiment reruns require private/local assets and are outside this closeout.
