# Kawaneen demo shot list

> Final portfolio relationship: The tracked final portfolio film is the
> approximately 72-second edited demo at `docs/demo/kawaneen-demo.mp4`. This
> document remains an extended/manual walkthrough reference and production
> history; it is not a description of the exact final edit.

The shot list matches the [three-minute script](three-minute-script.md) and is
intended for one clean recording pass.

1. **0:00–0:20 — problem/value:** product name, Arabic legal-document
   intelligence, evidence-first framing.
2. **0:20–0:40 — architecture:** show the two-profile Mermaid diagram. Make
   the full-local/private boundary and public synthetic/no-LLM boundary legible.
3. **0:40–1:10 — search:** public profile query `ما هي مدة الإرجاع؟`; show
   ranked evidence, result count, `KAWANEEN_DEMO` scope, document/source
   identity, exact passage, and provenance/chunk identity. Do not require hidden
   technical details in the main shot. If showing the full system locally, use
   an approved DEV query only.
4. **1:10–1:40 — grounded ask:** public profile query `ما هي مدة إشعار العقد؟`;
   show the grounded answer and citation card with document/article/page/quote
   plus the public-demo boundary. Canonical inspection is unavailable in the
   reduced public synthetic profile and must not be presented as succeeding.
   For a local full-system shot, show retrieval, evidence, answerability, and
   citation verification without exposing unnecessary private source text.
5. **1:40–2:00 — extraction:** paste
   `يلتزم الطرف بالسداد خلال ثلاثين يوماً.`; show the deterministic backend
   candidate `ثلاثين يوماً`, temporal candidate, normalized value `30 days`, and
   the empty semantic deadline field. The current public UI does not expose the
   raw candidate registry; keep disabled public upload/hybrid options and the
   input limit visible.
6. **2:00–2:25 — evaluation:** show tracked metrics, scope labels, retrieval
   evidence, Generation and Extraction sections, and technical provenance.
   If mentioning the 80/80 1.5B fallback-generator failure, briefly show the
   README or Phase-15 report where it is explicitly documented; do not imply
   that the polished Evaluation page newly surfaces it.
7. **2:25–2:45 — reproducibility:** show `docker compose up` or the full-local
   runbook, then `make phase16-verify`; keep the terminal free of private paths.
8. **2:45–3:00 — publication boundary:** show `make phase17-space-bundle`,
   `NOT_PUBLISHED_USER_APPROVAL_REQUIRED`, and finish on the synthetic/not-
   Saudi/not-legal-advice banner.

Use public synthetic data for public-facing interactions. Do not show private
corpus text, HOLDOUT, credentials, raw MLflow databases, personal
notifications, or a nonexistent live URL; do not imply a live deployment or
make a legal-advice claim. The tracked final portfolio film is complete and
validated at `docs/demo/kawaneen-demo.mp4`; this list remains extended/manual
recording history rather than the exact final edit.
