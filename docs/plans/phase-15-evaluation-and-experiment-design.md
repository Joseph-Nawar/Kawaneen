# Phase 15 — Evaluation and Experiment Report Design

**Project:** Kawaneen  
**Authoritative starting main:** `03f58284426c84c6c813be2b1e1bbbbbfd1c9a2d`  
**Status:** Approved design

## Goal

Build the repository's final research-evaluation layer by synthesizing frozen Phase 3–11 evidence, running only pre-registered DEV-only experiments needed to answer unresolved research questions, performing a human-reviewed error analysis, and producing a reproducible final experiment report without changing the production system.

## Research governance

Every material result must carry one provenance label:

- `HISTORICAL_FROZEN`
- `PHASE15_DEV`
- `HUMAN_REVIEWED_DIAGNOSTIC`

Protected Phase 3, Phase 8, and Phase 11 HOLDOUTs must not be rerun, regenerated, inspected for tuning, or used for Phase 15 selection. Existing one-shot HOLDOUT results may be cited only as frozen historical evidence.

Phase 6 labels remain described as externally AI-reviewed engineering labels, not human/expert gold. Negative and inconclusive findings must be reported.

## Final experiment matrix

| Experiment | Variants | Phase 15 treatment |
| --- | --- | --- |
| Normalization | raw, light, aggressive | Reuse Phase 4/7; new Arabic-embedding DEV sensitivity |
| Parsing | plain, layout-aware, OCR where relevant | Reuse Phase 3 only |
| Chunking | fixed-256, fixed-512, legal-structure, parent-child | Reuse Phase 5 |
| Retrieval | keyword, BM25, dense, hybrid | Reuse Phase 7–8 |
| Embeddings | multilingual-E5-small, BGE-M3, Arabic-Retrieval-v1.0 | first two frozen; Arabic model new DEV run |
| Reranking | none, BGE cross-encoder | reuse Phase 8 + pre-registered hard-query DEV slice |
| Query language | MSA, English, Egyptian, Gulf/Saudi, Levantine | existing Arabic/English + 60 new dialect variants |
| Generator | extractive, Qwen3-4B, ALLaM-7B 4-bit | extractive/Qwen frozen; ALLaM matched-80 DEV |
| Context | leaf/flat, neighbor, parent | reuse Phase 5 |
| Abstention | no score gate, score sensitivity, Stage-D | DEV-only counterfactual |
| Citation verification | pre-verification vs verified | persisted-output counterfactual |
| Latency | retrieval/generation latency-quality | fixed M5 protocol |

## Historical synthesis

Reuse, do not rerun:

- Phase 3 parsing qualification.
- Phase 4 normalization.
- Phase 5 chunking/context.
- Phase 7 keyword/BM25/dense/embeddings.
- Phase 8 hybrid/reranking.
- Phase 9 grounding.
- Phase 10 extractive/Qwen generation and forensics.
- Phase 11 relevant limitations.
- Phase 14 testing baseline.

Create a tracked, text-free evidence registry containing authoritative paths and hashes. Earlier frozen artifacts are read-only.

## Arabic embedding experiment

Pre-register exactly one new Arabic-specialized embedding model before any Kawaneen DEV score:

`omarelshehy/Arabic-Retrieval-v1.0`

Before scoring, lock exact Hugging Face revision, pooling, dimension, embedding normalization, prefixes/instructions, dtype, batch size, runtime/device, and model/config identities where practical.

Compare on the frozen Phase 7 DEV retrieval population:

- multilingual-E5-small;
- BGE-M3;
- Arabic-Retrieval-v1.0.

Metrics: Recall@1/5/10, MRR@10, nDCG@10, CompleteEvidenceRecall@5/10, p50/p95 latency.

Run the Arabic model under `arabic-raw-v1`, `arabic-light-v1`, and `arabic-aggressive-v1`. This sensitivity analysis cannot alter production normalization.

## Dialectal robustness experiment

Freeze exactly 20 MSA DEV base intents before paraphrase generation and before retrieval outcomes are seen. For each base intent create one:

- Egyptian Arabic paraphrase;
- Gulf/Saudi Arabic paraphrase;
- Levantine Arabic paraphrase.

Total: 60 dialect variants + 20 matched MSA controls.

Raw query text remains private. Paraphrases must preserve legal intent, dates, numbers, article/provision identifiers, and original qrels/evidence target. They must not narrow/broaden the question or add facts.

Describe the set as AI-generated, AI-validated diagnostic perturbations, not human-authored dialect gold.

Evaluate BM25, BGE-M3 dense, frozen hybrid, and frozen hybrid+reranker. Report paired dialect-minus-MSA Recall@10, MRR@10, nDCG@10, and CompleteEvidenceRecall@10 separately for Egyptian, Gulf/Saudi, and Levantine.

## Hard-query reranking analysis

Define and freeze a hard-query slice before computing reranker deltas. The rule may use query metadata and baseline properties such as multi-evidence, exact-provision, authority, deadline, cross-language, long-query bin, and low pre-rerank relevant rank. It must not use post-rerank gains.

Compare hybrid before/after BGE reranking with paired bootstrap CI, wins/ties/losses, and paired rank-biserial effect size where meaningful.

## Generator matched-80 experiment

Freeze exactly 80 DEV query IDs before ALLaM inference:

- 31 answerable with gold evidence present in Phase 8 top-8;
- 30 answerable with gold evidence absent from Phase 8 top-8;
- all 19 explicitly unanswerable.

Use deterministic seed `20260826` and balance language/category where possible. This is an enriched diagnostic subset and must not be presented as prevalence-representative.

Compare on the exact same query IDs and context blocks:

- deterministic extractive;
- Qwen3-4B-Instruct-2507;
- ALLaM-7B-Instruct-preview in a trustworthy 4-bit local form.

Reuse existing Qwen/extractive outputs where fingerprints match; otherwise rerun DEV only under frozen configuration. Never rerun HOLDOUT.

Metrics: SupportedAnswerPrecision, SupportedAnswerCoverage, FalseAnswerRate, FalseAbstentionRate, UnanswerableAbstentionRecall, CompleteGoldEvidenceUse, ValidCitationRate, GoldCitationHitRate, invalid-generation rate, successful verified answers, plus outcome taxonomy.

## ALLaM runtime policy for 16 GB M5

Full precision is forbidden. Before scoring:

1. lock the official `humain-ai/ALLaM-7B-Instruct-preview` revision;
2. identify a trustworthy 4-bit artifact/runtime whose provenance maps to official weights;
3. record quantization source/format/artifact SHA/runtime/context/output limits/disk footprint;
4. run a bounded M5 preflight.

If no trustworthy/stable 4-bit path exists, stop before scoring and pre-register one smaller credible Arabic-centric instruct fallback. Do not test multiple candidates against labels and then choose a winner.

Run large models sequentially.

## Citation verification counterfactual

Do not weaken serving. Offline, compare each persisted raw generator candidate response with a counterfactual "would surface without citation/support verification" state and the actual verifier outcome.

Report pre/post unsafe acceptance, paired absolute risk reduction + 95% CI, relative risk reduction where valid, coverage cost, and discordant pairs.

Apply to persisted Phase 10 outputs and new ALLaM matched-80 outputs.

## Abstention sensitivity

Production remains Stage-D. Run only a DEV-only uncalibrated reranker-score sensitivity analysis using score-distribution quantiles computed without relevance labels:

- no threshold;
- bottom 10% filtered;
- bottom 25% filtered;
- bottom 50% filtered.

Call this an `uncalibrated score-gate sensitivity analysis`. No threshold may be promoted to serving.

## Latency-quality protocol

On one fixed DEV latency subset, batch size 1, >=3 warmups, same top-k, measure keyword, BM25, E5 dense, BGE-M3 dense, Arabic-Retrieval dense, hybrid, and hybrid+reranker.

Report p50/p95 and quality. Main figure: nDCG@10 vs p50 latency.

For generators report p50/p95 generation latency, output token count, tokens/sec only if reliably exposed, disk footprint, and defensible process/unified-memory measurements only. Do not call RSS "GPU VRAM".

## Statistics

For new paired retrieval experiments use 2,000 deterministic paired bootstrap replicates, seed `20260826`, 95% CI, paired delta, wins/ties/losses, and paired rank-biserial where meaningful.

For paired binary generator/safety outcomes use risk difference, paired-bootstrap CI, discordant-pair counts, and exact McNemar only as a supplementary test when appropriate.

Latency reporting uses p50/p95, paired median difference, and latency ratio. Do not manufacture p-values.

## Assisted manual error analysis

Prepare approximately 120 failed/borderline DEV cases, with zero HOLDOUT cases. Candidate triggers should include retrieval misses/borderline ranks, retriever disagreement, reranker demotion, dialect degradation, false answers, false abstentions, malformed generation, quotation mismatch, support rejection, incomplete evidence, and citation rejection.

Stratify across language/register, pipeline stage, legal query category, answerability population, and severity.

The user must manually adjudicate at least 100 cases. Each gets one primary root-cause category, optional secondary, reviewer confidence, and optional note.

Required primary taxonomy:

1. OCR failure
2. article segmentation failure
3. normalization failure
4. lexical mismatch
5. semantic retrieval failure
6. reranker failure
7. missing source
8. wrong jurisdiction
9. insufficient context
10. generator hallucination
11. citation mismatch
12. ambiguous question

Use earliest demonstrable pipeline root cause as the primary label.

AI preclassification is assistance only. It must use a fixed local model/prompt and be hidden/collapsed by default in the UI to reduce anchoring. The human label is final.

After >=100 reviews report AI-human raw agreement, Cohen's kappa, and per-category agreement.

Final wording: `100+ cases manually adjudicated by one human reviewer, assisted by hidden-by-default AI preclassification.`

## Review application

Create a separate local Streamlit reviewer at `src/kawaneen/phase15/review_app.py`; do not integrate it into the Phase 13 product UI.

`make phase15-review` should open it. It must support persistent/atomic progress, restart/resume, immutable case IDs, `N / 120` progress, all taxonomy labels, optional secondary/confidence/note, and collapsed AI suggestion.

## Package boundary

Create `src/kawaneen/phase15/` with focused modules:

- `contracts.py`
- `statistics.py`
- `evidence.py`
- `selection.py`
- `embedding.py`
- `dialect.py`
- `reranking.py`
- `generation.py`
- `counterfactuals.py`
- `latency.py`
- `review.py`
- `review_app.py`
- `reporting.py`
- `orchestrator.py`

Reuse Phase 7–10 primitives rather than duplicating serving/retrieval logic.

## CLI and Make interface

Add `kawaneen phase15` subcommands:

`plan`, `freeze`, `synthesize`, `embedding`, `dialect-prepare`, `dialect-evaluate`, `reranking`, `generation-preflight`, `generation-run`, `counterfactuals`, `latency`, `review-prepare`, `review-status`, `finalize`.

`finalize` must refuse to run before >=100 unique human review decisions.

Recommended Make targets:

`phase15-freeze`, `phase15-synthesize`, `phase15-embedding`, `phase15-dialect`, `phase15-reranking`, `phase15-generation-preflight`, `phase15-generation`, `phase15-counterfactuals`, `phase15-latency`, `phase15-review-prepare`, `phase15-review`, `phase15-review-status`, `phase15-finalize`.

## Artifact layout

Private ignored root: `artifacts/private/phase15_evaluation/`.

Tracked sanitized/text-free artifacts:

- `data/manifests/evaluation/phase15_experiment_plan.json`
- `data/manifests/evaluation/phase15_evidence_registry.json`
- `data/manifests/evaluation/phase15_model_lock.json`
- `data/manifests/evaluation/phase15_dialect_manifest.json`
- `data/manifests/evaluation/phase15_generator_subset_manifest.json`
- `data/evaluation/phase15_embedding_metrics.json`
- `data/evaluation/phase15_dialect_metrics.json`
- `data/evaluation/phase15_reranking_metrics.json`
- `data/evaluation/phase15_generator_metrics.json`
- `data/evaluation/phase15_citation_counterfactual.json`
- `data/evaluation/phase15_abstention_sensitivity.json`
- `data/evaluation/phase15_latency_metrics.json`
- `data/evaluation/phase15_error_analysis.json` after human gate
- `data/evaluation/phase15_research_questions.json` after human gate
- `docs/reports/phase-15-evaluation-and-experiment-report.md` after human gate

## Figures

Generate reproducibly from sanitized metrics:

1. structure-aware vs fixed chunking effects;
2. retrieval quality vs latency;
3. MSA-to-dialect degradation;
4. generator support/coverage/safety comparison;
5. human-reviewed failure taxonomy;
6. optional citation-verification before/after.

No private text in figures.

## Research questions

Final report answers all seven with `SUPPORTED`, `PARTIALLY_SUPPORTED`, `INCONCLUSIVE`, or `NOT_SUPPORTED`:

1. Does structure-aware chunking outperform fixed chunks for Arabic legal retrieval?
2. How much does light Arabic normalization affect lexical and dense retrieval?
3. Does hybrid retrieval outperform dense-only retrieval across Arabic and English queries?
4. How much does reranking improve hard legal queries?
5. How robust is retrieval to dialectal paraphrasing?
6. Can citation verification meaningfully reduce unsupported answers?
7. How accurately can a zero-cost local model answer from Arabic legal evidence?

Each answer includes population, provenance, primary effect/CI, and limitations.

## Required limitations

Disclose that Phase 6 labels are AI-reviewed; Phase 15 manual analysis is one human engineering reviewer; dialect variants are AI-generated diagnostics; ALLaM matched-80 is enriched; ALLaM is quantized/hardware-specific; no tested statutory parser qualified for unattended v1 ingestion; historical HOLDOUTs were not rerun; and the work is an engineering evaluation, not proof of real-world legal correctness.

## Execution stages

A. Freeze plan/evidence/model/subset identities.  
B. Synthesize frozen Phase 3–11 evidence.  
C. Run embedding, dialect, hard-reranker, counterfactual, latency experiments.  
D. Run ALLaM matched-80 after preflight.  
E. Generate 120-case review packet and reviewer. **STOP for human review.**  
F. After >=100 human labels, aggregate, generate figures/RQ answers/final report.

## Completion gates

Phase 15 is complete only when:

- plan/model/subsets were frozen before outcomes;
- zero new protected HOLDOUT runs/access;
- exact Arabic embedding revision locked;
- 60 accepted dialect variants / 20 matched groups;
- exact matched-80 generator subset;
- ALLaM quantization provenance locked or fallback pre-registered before results;
- citation/abstention counterfactuals complete;
- latency-quality analysis complete;
- >=100 human decisions;
- paired CIs/effect sizes where practical;
- all seven RQs answered;
- every material claim traceable to evidence/artifact/hash;
- no private query/source text committed;
- public tests green, coverage >=85%, Ruff/Pyright green;
- production architecture unchanged;
- frozen Phase 3–14 results unchanged.

## Hard prohibitions

No protected HOLDOUT reruns; no tuning on HOLDOUT; no production model/policy/chunk/retrieval change; no model shopping after DEV outcomes; no human-review fabrication; no human-gold claim; no external upload of private legal/query text; no fake leaderboard across unmatched populations; no suppression of negative/inconclusive results.
