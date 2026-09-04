# Kawaneen demo shot list

The shot list matches the [three-minute script](three-minute-script.md) and is
intended for one clean recording pass.

1. **0:00–0:20 — problem/value:** product name, Arabic legal-document
   intelligence, evidence-first framing.
2. **0:20–0:40 — architecture:** show the two-profile Mermaid diagram. Make
   the full-local/private boundary and public synthetic/no-LLM boundary legible.
3. **0:40–1:10 — search:** public profile query `ما هي مدة الإرجاع؟`; show
   five-result cap, synthetic document identity, RRF metadata, and exact
   evidence. If showing the full system locally, use an approved DEV query only.
4. **1:10–1:40 — grounded ask:** public profile query `ما هي مدة إشعار العقد؟`;
   expand `E001` and show the exact quote. For a local full-system shot, show
   retrieval, evidence, answerability, and citation verification without
   exposing unnecessary private source text.
5. **1:40–2:00 — extraction:** paste
   `يلتزم الطرف بالسداد خلال ثلاثين يوماً.`; show deterministic candidate/span,
   disabled public upload/hybrid options, and the input limit.
6. **2:00–2:25 — evaluation:** show headline links and the negative 1.5B
   fallback result; keep inconclusive/partial labels visible.
7. **2:25–2:45 — reproducibility:** show `docker compose up` or the full-local
   runbook, then `make phase16-verify`; keep the terminal free of private paths.
8. **2:45–3:00 — publication boundary:** show `make phase17-space-bundle`,
   `NOT_PUBLISHED_USER_APPROVAL_REQUIRED`, and finish on the synthetic/not-
   Saudi/not-legal-advice banner.

Do not show private corpus text, HOLDOUT, credentials, raw MLflow databases,
personal notifications, or a nonexistent live URL. The video is not complete
until the real file exists at `docs/demo/kawaneen-demo.mp4` and has been
validated.
