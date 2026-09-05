# Portfolio Final Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the frozen Kawaneen system as a recruiter-facing portfolio repository without changing research, model, retrieval, extraction, deployment, or evaluation behavior.

**Architecture:** Keep the existing implementation and historical phase record intact. Add a thin public navigation layer consisting of a concise README, public system/component cards, consolidated safety and provenance evidence, and explicit demo/deployment pointers. Derive every metric, revision, and status from tracked artifacts; keep private MLflow and corpus material local and ignored.

**Tech Stack:** Markdown, JSON, YAML/CFF, existing Python/uv tooling, GitHub CLI if authenticated, and existing rendered repository assets.

**Spec:** User-provided final portfolio closeout brief supplied with this task.

## Global Constraints

- Preserve the authoritative baseline `e60ef489020cd167728f31d1a3ded4b23bcc8616`.
- Work only on `portfolio/final-closeout` in the isolated worktree.
- Do not access HOLDOUT, rerun experiments, tune models, alter frozen results, publish Hugging Face, purchase anything, or merge.
- Use Python `>=3.11,<3.13` and `uv`; run the specified quality, regression, E2E, Phase 17, link, and security checks.
- Do not commit private data, raw legal text, credentials, local paths, MLflow databases, or generated junk.
- The demo video remains `MANUAL VIDEO PENDING` until a real `docs/demo/kawaneen-demo.mp4` exists and validates.

---

### Task 1: Establish authoritative evidence map

**Files:**
- Read: `docs/reports/phase-15-evaluation-and-experiment-report.md`, `docs/reports/phase-16-observability-and-reproducibility.md`, Phase 17 reports/docs, `data/evaluation/*`, `data/manifests/*`, `docs/architecture/*`, `docs/deployment/*`, `docs/demo/*`
- Create: `docs/superpowers/plans/2026-09-04-portfolio-final-closeout.md`

- [ ] Extract only tracked, public-safe metric names/values, model identities, hashes, commands, and limitations for use in the portfolio documents.
- [ ] Confirm existing screenshots and architecture source are representative and identify the best top-of-README visual.
- [ ] Confirm local MLflow database/artifacts are ignored and identify safe existing reproducibility metadata for a text-only export.

### Task 2: Write recruiter-facing landing page and navigation

**Files:**
- Modify: `README.md`
- Create: `docs/README.md`
- Create if feasible: `docs/assets/portfolio/social-preview.png`

- [ ] Reframe the README around the problem, implemented system, evidence, demo, local run paths, limitations, and documentation map rather than phase chronology.
- [ ] Add only defensible badges, 3–5 linked headline results, one strong existing architecture/UI visual, and a clearly pending video section.
- [ ] Add recruiter-oriented first-level links and preserve accurate claims about the full local system versus the synthetic public demo.
- [ ] Keep the social preview optional and limited to a clean 1280×640 static design; do not install a large graphics stack.

### Task 3: Add public system, data, safety, and observability evidence

**Files:**
- Create: `docs/model-card.md`
- Create: `docs/dataset-card.md`
- Create: `docs/safety-and-limitations.md`
- Create: `docs/mlflow-evidence.md`
- Create: `data/evaluation/portfolio_mlflow_evidence.json`
- Create only if needed: one small deterministic export script under `scripts/`

- [ ] Document components, frozen revisions/configuration, pretrained-versus-custom boundaries, evaluation evidence, negative fallback-generator result, and failure modes without claiming end-to-end legal correctness.
- [ ] Distinguish the private canonical corpus, frozen DEV/HOLDOUT assets, and synthetic `KAWANEEN_DEMO` corpus, including governance and redistribution boundaries.
- [ ] Consolidate evidence-first safety, jurisdiction, abstention, invalid-output handling, latency, licensing, label, and public-demo limitations.
- [ ] Export only safe MLflow metadata already present locally; explicitly exclude queries, legal/evidence/generation/extraction text, sensitive paths, query hashes, and stack traces.
- [ ] Validate the export schema and ensure it does not access HOLDOUT or private content.

### Task 4: Finish repository metadata and completion checklist

**Files:**
- Create: `docs/portfolio-completion.md`
- Create: `SECURITY.md`
- Create: `CITATION.cff`
- Modify via GitHub CLI if authenticated: repository description and topics only; leave visibility and homepage unchanged/blank

- [ ] Map every original roadmap criterion to an exact artifact with `COMPLETE` or `MANUAL VIDEO PENDING`, leaving only the manual video criterion pending.
- [ ] Add truthful private-reporting guidance without inventing an email address or support promise.
- [ ] Add CFF metadata from repository identity/version/license without inventing DOI, affiliation, or ORCID.
- [ ] Record final repository metadata in the draft PR description.

### Task 5: Polish the one-pass demo recording plan

**Files:**
- Modify: `docs/demo/three-minute-script.md`
- Modify: `docs/demo/shot-list.md`

- [ ] Specify the 0:00–3:00 sequence, exact full-system versus synthetic-demo views, safe representative data, visible limitations, and commands needed to prepare each screen.
- [ ] Include the exact destination `docs/demo/kawaneen-demo.mp4`, recording safety requirements, and the available ffmpeg compression command.
- [ ] Do not fabricate or add an MP4 during this checkpoint.

### Task 6: Audit, validate, commit, and open one draft PR

**Files:**
- All changed files from Tasks 2–5

- [ ] Run `uv run ruff format --check .`, `uv run ruff check .`, `uv run pyright`, `make check`, `make test-regression`, `make test-e2e`, `make phase17-verify`, and `git diff --check`.
- [ ] Validate all new relative Markdown links, CFF syntax, MLflow evidence JSON, no absolute/private paths or secrets, and `git ls-files artifacts/private` is empty.
- [ ] Review the landing page as a 60-second recruiter scan and a technical ML-engineer scan; fix navigation gaps without adding historical churn.
- [ ] Commit the portfolio closeout, push `portfolio/final-closeout`, and create exactly one draft PR against `main`; never merge.
- [ ] Return the branch, HEAD, PR, base SHA, files, evidence sources, tests, governance status, and exact manual recording instructions; stop with `VIDEO_MANUAL_RECORDING_REQUIRED`.

### Self-review checklist

- [ ] Every closeout section A–T in the brief is covered by a task above.
- [ ] No new metric or model claim lacks an authoritative tracked source.
- [ ] No task changes runtime behavior or touches private/HOLDOUT data.
- [ ] The only intentionally unfinished product action after this checkpoint is recording the real MP4; Hugging Face publication remains optional and unpublished.
