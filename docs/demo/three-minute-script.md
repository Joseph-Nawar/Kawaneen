# Three-minute Kawaneen demo script

Use the public demo UI with `KAWANEEN_UI_MODE=live`,
`KAWANEEN_UI_PUBLIC_DEMO=true`, and the demo FastAPI service. The script is
exactly 3:00; all text shown is fictional and synthetic.

| Time | Narration | Screen/action | Expected visible result |
| --- | --- | --- | --- |
| 0:00–0:25 | “Kawaneen makes Arabic legal research inspectable: retrieve evidence, preserve source identity, and abstain when evidence is insufficient.” | Open the landing/search screen. Point to the `PUBLIC DEMO PROFILE` banner. | Banner states synthetic corpus, reduced retrieval, no generative answer, not legislation, not advice. |
| 0:25–0:50 | “The full local profile uses the frozen BGE hybrid stack, exact Qdrant search, Ollama Stage-D, verifier, and MLflow. This public profile removes private data and generation.” | Show `docs/architecture/phase17-deployment.mmd` or its rendered Mermaid preview. | Both deployment profiles and the no-LLM/no-Qdrant/no-MLflow/no-Ollama demo boundary are visible. |
| 0:50–1:25 | “Search is retrieval-first. I’ll ask a fictional question about returns.” | Search: `ما هي مدة الإرجاع؟`; click Search; open one evidence card. | Up to five ranked passages from `KAWANEEN_DEMO`, with RRF metadata and exact synthetic disclaimer. |
| 1:25–1:55 | “Ask does not invent a legal conclusion. It presents the exact top passage with a citation.” | Open Ask; enter `ما هي مدة إشعار العقد؟`; click Ask; expand the citation. | Answer is an exact synthetic passage, with `KAWANEEN_DEMO` source identity and no model-generated prose. |
| 1:55–2:15 | “Extraction stays deterministic and bounded.” | Open Extract; confirm only Paste text; paste `يلتزم الطرف بالسداد خلال ثلاثين يوماً.`; click Extract. | Candidate/deadline span appears; upload and Hybrid options are absent; limits are visible. |
| 2:15–2:40 | “The evaluation view separates tracked evidence from live latency and keeps negative findings visible.” | Open Evaluation. | Tracked aggregate cards/links are shown; the Phase 15 fallback-generation failure is not hidden. |
| 2:40–3:00 | “The full stack runs with `docker compose up`; the demo exports deterministically and remains unpublished until approval. This is evidence-first research infrastructure, not legal advice.” | Show terminal with `make phase17-space-bundle` and the full-local docs; finish on banner. | Export manifest and publication gate are visible; end on synthetic/not-advice disclaimer. |
