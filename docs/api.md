# Kawaneen v1 API

The production boundary is served with:

```bash
uv run kawaneen api serve
# or
make api-serve
```

The default bind is `127.0.0.1:8000`. OpenAPI is available at `/docs` and `/openapi.json`.
Version 1 accepts only `jurisdiction: "SA"`.

## Endpoints

- `POST /v1/search`: hybrid sparse top-50 + dense top-50, frozen RRF, 20 candidates, reranker, and serving depth up to 8. Returned scores are raw reranker logits with `score_type: "reranker_raw_logit"`.
- `POST /v1/answer`: one-query retrieval, Phase 9 grounding, Phase 10 answerability gates, Stage-D generation when permitted, and exact verified citations. A refusal is HTTP 200 with `answerable: false`, `answer: null`, and `abstention_reason`.
- `POST /v1/extract`: deterministic candidate extraction or the limited hybrid Phase 11 assembly. Request text is never persisted and is identified as `api-request` / `request-body` provenance.
- `GET /v1/documents`: deterministic document-ID order with `offset`, `limit` (1–100, default 20), and `total`.
- `GET /v1/documents/{document_id}`: safe metadata and canonical units/text. Local paths are never returned.
- `GET /v1/health`: readiness only; it never runs inference. Required-component degradation returns HTTP 503.
- `GET /v1/models`: configured capability/provider/model/revision/loaded/ready metadata; it never triggers loading.

## Limits and errors

Queries are capped at 2,000 characters, extraction text at 20,000 characters, search limit at 8, and POST bodies at 128 KiB. Clients may send a safe `X-Request-ID` up to 128 characters; invalid or missing IDs are replaced with a UUID and echoed in the response header and body.

Errors have the following shape and never contain stack traces, prompts, provider responses, filesystem paths, or private artifact details:

```json
{
  "error": {"code": "DOCUMENT_NOT_FOUND", "message": "document not found"},
  "request_id": "..."
}
```

Stable codes are `VALIDATION_ERROR` (422), `REQUEST_TOO_LARGE` (413), `DOCUMENT_NOT_FOUND` (404), `SERVICE_UNAVAILABLE` / `MODEL_UNAVAILABLE` (503), `REQUEST_TIMEOUT` (504), and `INTERNAL_ERROR` (500).

Synchronous retrieval and model work runs in worker threads with configurable defaults of 10 seconds for search, 65 seconds for answer, and 35 seconds for extraction. Request-scoped logs carry only the request ID; complete query and extraction text are not logged.
