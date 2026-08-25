# Phase 15 Evaluation and Experiment Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, provenance-aware Phase 15 evaluation layer that synthesizes frozen evidence, runs pre-registered DEV-only experiments, creates a 120-case assisted human review workflow, and finalizes the research report only after >=100 human adjudications.

**Architecture:** Use a dedicated offline `kawaneen.phase15` package. Historical artifacts are read-only. New raw results live under ignored `artifacts/private/phase15_evaluation/`; tracked outputs are sanitized/text-free. Production API/UI/serving selections and protected HOLDOUTs are unchanged.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, NumPy, existing retrieval/generation components, sentence-transformers retrieval extra, Hugging Face Hub, Streamlit local reviewer, pytest, Ruff, Pyright.

**Spec:** `docs/plans/phase-15-evaluation-and-experiment-design.md`

## Global constraints

- Base exactly `03f58284426c84c6c813be2b1e1bbbbbfd1c9a2d`.
- Work on isolated branch/worktree, not `main`.
- No new Phase 3/8/11 HOLDOUT runs or tuning access.
- No frozen Phase 3–14 artifact mutation.
- New experiments are DEV/diagnostic only.
- No production promotion/change.
- Arabic embedding candidate fixed to `omarelshehy/Arabic-Retrieval-v1.0` before scoring.
- ALLaM 7B is 4-bit local only on M5 16 GB.
- No external upload of private query/source text.
- Public CI stays model-download-free/private-data-free.
- Coverage remains >=85%; Ruff/Pyright/Python 3.11/3.12 remain green.
- Stage E stops for real human review; finalization hard-fails before >=100 labels.
- TDD and bounded commits.

## File map

Create `src/kawaneen/phase15/` with `contracts.py`, `statistics.py`, `evidence.py`, `selection.py`, `embedding.py`, `dialect.py`, `reranking.py`, `generation.py`, `counterfactuals.py`, `latency.py`, `review.py`, `review_app.py`, `reporting.py`, `orchestrator.py`.

Modify `src/kawaneen/cli.py`, `Makefile`, `.gitignore` if necessary, and `pyproject.toml` only when a dependency is genuinely required. Do not add an Apple-only ALLaM runtime to public cross-platform dependencies.

Add `tests/phase15/` unit tests. Private/model tests use existing `private_artifact` / `model_artifact` markers.

## Task 1 — Contracts

Create strict enums/Pydantic models for provenance, RQ status, the 12 error categories, experiment plan, model locks, dialect manifest, generator subset, review decision, and metric summaries.

TDD must reject unknown provenance/categories, invalid bootstrap settings, generator subset counts not equal to 31/30/19, and review decisions lacking a primary category.

Commit: `feat: add Phase 15 research contracts`

## Task 2 — Statistics

Implement deterministic:

- `paired_bootstrap_delta(..., seed, replicates=2000, confidence=.95)`
- `paired_risk_difference(...)`
- `paired_rank_biserial(...)`
- `cohens_kappa(...)`

Tests cover deterministic repeatability, all ties, all wins, empty/mismatched inputs, perfect kappa, and zero-delta handling.

Commit: `feat: add Phase 15 paired statistics`

## Task 3 — Experiment freeze + evidence registry

Implement `kawaneen phase15 plan` and `freeze`, plus `make phase15-freeze`.

Freeze seed `20260826`, 2000 bootstrap replicates, seven RQs, hard prohibitions, pre-registered Arabic embedding ID, base SHA, and a read-only historical evidence registry with hashes for Phase 3/4/5/6/7/8/9/10/11/14.

Tests verify every registered tracked hash and that no HOLDOUT private path is part of the registry.

Commit: `feat: freeze Phase 15 experiment governance`

## Task 4 — DEV subset freeze

Implement deterministic selection before new model results:

- 20 unique MSA base intents for dialect perturbations;
- generator subset exactly 31 gold-present answerable + 30 gold-absent answerable + all 19 unanswerable = 80;
- deterministic 120-case review candidate selector schema, though final packet may only be instantiated after new experiment failures exist.

Tracked manifests contain hashes/counts/distributions only, no query text.

Commit: `feat: freeze Phase 15 DEV subsets`

**Hard gate:** do not revise these subsets because of later results.

## Task 5 — Arabic embedding pre-registration and DEV evaluation

Public tests use fake encoders and prove exact model ID enforcement, identical query/qrel identities across normalization variants, no HOLDOUT loader, and lock immutability.

Resolve exact HF revision for `omarelshehy/Arabic-Retrieval-v1.0` and commit the lock before scoring. Record pooling, dimension, normalization/prefix contract, dtype, batch size, package/runtime/device.

Then evaluate DEV only against existing E5 and BGE-M3 using existing Phase 7 metric definitions. Run raw/light/aggressive sensitivity for the Arabic model. Compute 2000 paired bootstrap CIs and p50/p95 latency. Write aggregate `phase15_embedding_metrics.json`.

Commit: `exp: evaluate Arabic retrieval embedding on DEV`

## Task 6 — Dialect packet + evaluation

Using the frozen 20 MSA IDs, generate one Egyptian, Gulf/Saudi, and Levantine paraphrase each with a fixed local model/prompt. Validate semantic equivalence before any retrieval results are inspected. Article numbers/dates/numeric identifiers must be preserved.

Private raw packet under `artifacts/private/phase15_evaluation/dialect/`; tracked manifest only hashes/counts.

Evaluate BM25, BGE-M3, hybrid, hybrid+reranker. Compute matched dialect-minus-MSA Recall@10, MRR@10, nDCG@10, CompleteEvidenceRecall@10 by dialect and pooled.

Commit: `exp: evaluate dialectal Arabic retrieval robustness`

## Task 7 — Hard-query reranking

Freeze a deterministic hard-query rule that cannot depend on reranker gains. Reuse existing DEV Phase 8 per-query artifacts; do not rerun HOLDOUT.

Compute paired bootstrap deltas, wins/ties/losses, rank-biserial. Write `phase15_reranking_metrics.json`.

Commit: `exp: analyze reranking on hard legal queries`

## Task 8 — Citation + abstention counterfactuals

Implement offline citation-verifier counterfactual from persisted Phase 10 outcomes, later extended with ALLaM outputs. Report pre/post unsafe acceptance, paired absolute risk reduction + CI, relative risk reduction where valid, coverage cost, discordant pairs.

Implement score-gate sensitivity with quantile thresholds derived from scores only: none, bottom10, bottom25, bottom50. Explicitly call it uncalibrated. No serving changes.

Commit: `exp: add citation and abstention counterfactuals`

## Task 9 — Latency-quality protocol

Implement fixed batch-1, >=3-warmup p50/p95 measurement with testable injected monotonic clock. Run on one fixed DEV subset for keyword/BM25/E5/BGE/Arabic embedding/hybrid/reranked on M5. Record machine/runtime identity and aggregate results.

Commit: `exp: measure Phase 15 latency quality tradeoffs`

## Task 10 — ALLaM 4-bit preflight and lock

Public tests with fake runtime must reject full precision, reject wrong base model, require exact official revision + quantization artifact hash, prohibit fallback after ALLaM outcomes exist, and enforce exact matched-80 IDs/contexts.

Resolve official `humain-ai/ALLaM-7B-Instruct-preview` revision and a trustworthy provenance-linked 4-bit local artifact/runtime. Record quantization source/format/hash/runtime/context/output limits/disk footprint. Run bounded M5 preflight.

If no trustworthy/stable path passes, stop before scoring and pre-register one smaller credible Arabic-centric fallback. Do not test multiple candidates against labels.

Commit the model/runtime lock before scoring: `exp: lock Phase 15 Arabic generator runtime`.

## Task 11 — Matched-80 generator run

Enforce identical query/context identities across extractive, Qwen, ALLaM. Reuse existing extractive/Qwen outputs if fingerprints match; otherwise rerun DEV only under frozen Phase 10 configuration. Run ALLaM sequentially on the exact same contexts.

Apply the same citation/support verification path. Compute the approved generator metrics and paired CIs, outcome taxonomy, and generator latency. Extend citation counterfactual with ALLaM.

Commit: `exp: compare local Arabic legal generators`

## Task 12 — 120-case assisted human review workflow

Create deterministic private packet of 120 failed/borderline DEV cases, zero HOLDOUT, stratified across language, stage, legal category, answerability, severity. Use all 12 approved root-cause categories and earliest-failure precedence.

Implement fixed local AI preclassification separately from final human label. In `review_app.py`, AI suggestion must be collapsed/hidden by default.

Reviewer UI must support query/evidence/diagnostics, primary/secondary/confidence/note, previous/next/save, persistent atomic resume, immutable IDs, and progress `N / 120`.

Add:

- `make phase15-review-prepare`
- `make phase15-review`
- `make phase15-review-status`

Tests must prove exactly 120 unique IDs, zero HOLDOUT IDs, atomic progress, duplicate updates not double-counted, and finalization rejection below 100.

Commit: `feat: add Phase 15 assisted error review`

### Mandatory human checkpoint

STOP HERE. Do not create final error-analysis/RQ/report artifacts. Do not invent human labels.

Return branch/HEAD, experiment results so far, exact model locks, M5 ALLaM preflight, exact 80 counts, 120 review-packet distributions, exact review command/private progress path/current count, tests/coverage, and integrity confirmation. Do not merge.

## Task 13 — Human review aggregation (continuation only after >=100)

Validate >=100 unique decisions. Aggregate error category × language/pipeline/query type/generator/severity. Compute AI-human raw agreement, Cohen's kappa, per-category agreement. Write text-free `phase15_error_analysis.json`.

Commit: `research: aggregate Phase 15 human error analysis`

## Task 14 — Figures + seven RQs

Generate at least five reproducible figures from sanitized metrics: structure-vs-fixed, quality-vs-latency, dialect degradation, generator safety/coverage, human taxonomy; optional verifier before/after.

Create `phase15_research_questions.json` with all seven questions, allowed status, population, provenance, effect/CI, limitation.

Commit: `research: synthesize Phase 15 research questions`

## Task 15 — Final report

Create `docs/reports/phase-15-evaluation-and-experiment-report.md` only after >=100 human labels. Contract tests must require full experiment matrix, all seven RQs, statistical methodology, human review count, error taxonomy, latency discussion, limitations, provenance labels, and forbid human-gold/HOLDOUT-rerun claims.

Cross-check every numeric claim to tracked evidence/artifact hashes.

Commit: `docs: add Phase 15 evaluation report`

## Task 16 — Final verification + PR

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
make check
make test-regression
git diff --check
```

Also run relevant private/model Phase 15 validation commands. Verify coverage >=85%, `git ls-files artifacts/private` returns none, no protected HOLDOUT access was introduced, and frozen Phase 3–14 result artifacts are unchanged from base.

Push and open PR against main. Require fresh exact-head Python 3.11/3.12 + existing Compose E2E. Do not merge.
