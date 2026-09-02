# Public demo deployment

The public demo is intentionally not production-equivalent. It uses:

- a fictional `KAWANEEN_DEMO` corpus of 64 project-created Arabic passages;
- `intfloat/multilingual-e5-small` at the locked revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`, with bundled corpus vectors;
- BM25, exact NumPy dense search, and small RRF depths (12/12/8);
- at most five evidence passages and deterministic exact-passage output;
- deterministic extraction only, with an 8,000-character input limit;
- no LLM generation, Qwen, Ollama, Qdrant, MLflow, private corpus, or Saudi
  source text.

Every displayed proposition is an exact passage from the synthetic corpus. An
insufficient match abstains rather than fabricating a legal conclusion. The
banner says `PUBLIC DEMO PROFILE`, and the live UI uses the FastAPI service;
the old fixture-only UI mode is not silently substituted.

## Build the unpublished bundle

```bash
make phase17-space-bundle
```

The ignored `build/phase17-space/` directory contains one Docker Space
container: FastAPI on loopback `8000` and Streamlit on public port `7860`.
It contains no `.git`, `.env`, credentials, private artifacts, full corpus,
Ollama state, Qdrant state, or MLflow database. Publication status remains
`NOT_PUBLISHED_USER_APPROVAL_REQUIRED`.

## Limits

Queries are limited to 500 characters, evidence to five results, extraction
input to 8,000 characters, concurrent AI/retrieval work to one request, and a
fixed-window budget of approximately 30 requests per minute. File uploads are
disabled in public-demo UI mode.

The bundled E5 model may need to be fetched by the hosting runtime for query
embedding; corpus vectors are precomputed and bundled. No generator is
downloaded or packaged.
