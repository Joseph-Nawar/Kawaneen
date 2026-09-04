# API examples

These examples use the current Phase 12 contracts. Replace the base URL only
when using the public Space or another local service.

```bash
BASE=http://127.0.0.1:8000

curl -fsS "$BASE/v1/health"
curl -fsS "$BASE/v1/models"

curl -fsS -X POST "$BASE/v1/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"ما هي مدة الاعتراض؟","jurisdiction":"SA","limit":5}'

curl -fsS -X POST "$BASE/v1/answer" \
  -H 'Content-Type: application/json' \
  -d '{"query":"ما هي مدة الاعتراض؟","jurisdiction":"SA"}'

curl -fsS -X POST "$BASE/v1/extract" \
  -H 'Content-Type: application/json' \
  -d '{"text":"يلتزم الطرف بالسداد خلال ثلاثين يوماً.","jurisdiction":"SA","mode":"deterministic"}'
```

The full profile returns frozen hybrid retrieval metadata and may issue a
verified Stage-D answer. The public profile returns `strategy:
"demo_retrieval_first"`, `score_type: "rrf_score"`, and an answer that is an
exact synthetic evidence passage; it never invokes a generator. Errors use
the normal `{ "error": { "code", "message" }, "request_id" }` envelope.

For the public synthetic profile, use `KAWANEEN_DEMO` instead of `SA`:

```bash
curl -fsS -X POST "$BASE/v1/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"ما هي مدة الاعتراض؟","jurisdiction":"KAWANEEN_DEMO","limit":5}'
```
