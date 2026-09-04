# Three-minute Kawaneen demo script

This is a one-pass recording plan for the real finished project. Use the full
local profile for the architecture/deployment claim and a safe DEV view only
when its evidence is approved for local recording. Use the public synthetic
profile for every public-facing data view unless the recording is strictly
local and the private corpus is intentionally visible.

## Before recording

### Full local profile (local-only evidence)

Keep the private artifact root outside Git and mount it read-only:

```bash
export KAWANEEN_HOST_ARTIFACTS_DIR=/absolute/path/to/private-artifacts
docker compose up --build
```

Use a representative approved DEV query/evidence view, never HOLDOUT. Do not
show private paths, raw source text beyond what is necessary, credentials, or
personal notifications. The exact full-local runbook is
[`docs/deployment/full-local.md`](../deployment/full-local.md).

### Public synthetic profile (preferred for sharing)

The synthetic UI/API profile is fictional and has no generator:

```bash
make phase17-space-bundle

# terminal 1: synthetic FastAPI service
uv run uvicorn kawaneen.demo.runtime:create_demo_app --factory \
  --host 127.0.0.1 --port 8000

# terminal 2: live public-demo UI
KAWANEEN_UI_MODE=live KAWANEEN_UI_PUBLIC_DEMO=true \
  KAWANEEN_API_URL=http://127.0.0.1:8000 \
  uv run streamlit run src/kawaneen/ui/app.py
```

For the actual prepared container, build/run the ignored bundle according to
the generated manifest. End on the persistent `PUBLIC DEMO PROFILE` banner.

## Exact 3:00 recording

| Time | Narration | Screen/action | Required visible result |
| --- | --- | --- | --- |
| 0:00–0:20 | “Kawaneen makes Arabic legal research inspectable: retrieve evidence, preserve source identity, and abstain when evidence is insufficient.” | Open the product landing/search view. | Project name, Arabic legal-intelligence purpose, and evidence-first framing. |
| 0:20–0:40 | “The full local profile combines BM25, dense retrieval, reranking, grounded generation, citation verification, and MLflow traces. The public profile removes private data and generation.” | Show the rendered `docs/architecture/phase17-deployment.mmd` diagram or a clean editor preview. | Both profiles; demo boundary clearly says synthetic, no LLM, no Qdrant, no MLflow, no Ollama. |
| 0:40–1:10 | “Search is the first inspection step. I’ll ask a fictional question about returns.” | On the public profile, search `ما هي مدة الإرجاع؟`; open one result. On a local full profile, use only an approved DEV query. | Ranked evidence, source identity, jurisdiction label, and exact passage; no private or HOLDOUT text in a shared recording. |
| 1:10–1:40 | “Ask is grounded in retrieved evidence and can abstain. It does not turn unsupported text into a confident legal conclusion.” | Public profile: ask `ما هي مدة إشعار العقد؟`; expand the citation. Full profile: show retrieval → context → verified citation if locally approved. | Exact synthetic passage for public mode, or verified local citation; show the boundary/disclaimer. |
| 1:40–2:00 | “Extraction stays deterministic and bounded.” | Public profile: paste `يلتزم الطرف بالسداد خلال ثلاثين يوماً.` into Extract and run it. | Candidate/deadline span; upload and hybrid controls absent in public mode; limits visible. |
| 2:00–2:25 | “The evaluation record keeps both measured gains and failures visible.” | Open Evaluation; show headline result links and the Phase 15 fallback-generator failure. | Positive/inconclusive results are labelled by scope; the 80/80 invalid fallback result is not hidden. |
| 2:25–2:45 | “The full system is reproducible locally with Docker Compose, while MLflow remains optional and local.” | Show a clean terminal with `docker compose up`/the full-local runbook, then `make phase16-verify`. | Deployment command, tracked identity/result reconstruction, and no secret/private path. |
| 2:45–3:00 | “The public demo is synthetic and unpublished. This is evidence-first research infrastructure, not legal advice.” | Show `make phase17-space-bundle`, its publication gate, then finish on the public-demo banner. | `NOT_PUBLISHED_USER_APPROVAL_REQUIRED`, synthetic/not-Saudi/not-advice disclaimer. |

## Recording constraints

- Record at 1080p where possible, 720p minimum; keep text readable.
- Do not show notifications, personal windows, private paths, secrets, raw
  customer/legal material, or HOLDOUT content.
- No copyrighted background music is required.
- The target file is `docs/demo/kawaneen-demo.mp4`.
- If `ffmpeg` is available, compress a valid recording with:

```bash
ffmpeg -i input.mp4 -vf "scale=-2:1080" -c:v libx264 -preset medium -crf 25 \
  -c:a aac -b:a 128k -movflags +faststart docs/demo/kawaneen-demo.mp4
```

If the source is already 720p, replace `scale=-2:1080` with
`scale=-2:720`. Keep the final file below GitHub’s normal per-file limit and
inspect its duration, streams, resolution, and size before adding it.
