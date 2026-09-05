# Kawaneen demo shot list

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
   show a numbered citation such as `01 · <document title>`, expand `Inspect
   citation 1`, and show the exact quote. For a local full-system shot, show
   retrieval, evidence, answerability, and citation verification without
   exposing unnecessary private source text.
5. **1:40–2:00 — extraction:** paste
   `يلتزم الطرف بالسداد خلال ثلاثين يوماً.`; show deterministic candidate/span,
   disabled public upload/hybrid options, and the input limit.
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
make a legal-advice claim. The video is not complete
until the real file exists at `docs/demo/kawaneen-demo.mp4` and has been
validated.
