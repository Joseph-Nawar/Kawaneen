# Kawaneen Phase 15 evaluation and experiment report

## Executive summary

Phase 15 is a research synthesis and pre-registered DEV evaluation. It did not
change production retrieval, chunking, generation, API, UI, or Stage-D policy.
Historical evidence is marked `HISTORICAL_FROZEN`; new measurements are marked
`PHASE15_DEV`; and the final 30-case diagnostic is marked
`AUTOMATED_ADJUDICATION_DIAGNOSTIC`.

The main results are mixed. Structure-aware chunking was better than the fixed
256-token baseline on the frozen Phase 5 challenge. The corrected hard-query
reranker slice showed small positive paired effects. The preregistered Arabic
embedding ran locally, but its raw normalization variant was not uniformly
better than BGE-M3 and normalization effects were not stable across historical
and new measurements. The corrected dialect analysis is inconclusive after a
content-only audit removed one malformed concatenated variant. Citation
verification removed the measured DEV contract-defect exposure from 29/40 to
0/40. The locked 1.5B Arabic fallback failed the frozen generation contract on
all 80 matched cases; ALLaM was blocked before scoring because no trustworthy
local 4-bit provenance path was available.

The diagnostic audit contains 23 confirmed operational failures, 7 borderline
cases, and 0 uncertain cases. Six confirmed output-contract failures are
reported separately from the 12-category root-cause taxonomy. This is not human
review, expert review, legal review, human gold, or a population-prevalence
estimate.

## Governance and provenance

The exact Phase 15 base is `03f58284426c84c6c813be2b1e1bbbbbfd1c9a2d` with seed
`20260826`, 2,000 paired-bootstrap replicates, and 95% intervals. The read-only
evidence registry hashes the frozen Phase 3, 4, 5, 6, 7, 8, 9, 10, 11, and 14
tracked artifacts: `data/manifests/evaluation/phase15_evidence_registry.json`.

The protocol amendment is recorded chronologically in
`docs/reports/phase-15-protocol-amendment.md`. The original plans remain the
historical design record. No Phase 3, Phase 8, or Phase 11 HOLDOUT was accessed
or rerun. Historical HOLDOUT metrics, where cited by the repository's prior
reports, remain frozen aggregates only.

Phase 6 data is externally AI-reviewed engineering data, not human gold or
expert gold. Phase 15's 120-case diagnostic population has hash
`8bc039f51344f3af47b817a5e1bbf51d4d087f49768ea5ca2b2ce3a29cd53777`; the frozen
30-case audit subset has hash
`4fc44ab5f5284ed720421dd13ef6f866e69292b5709b753c5464afacc6bd8af9`. No human
labels were collected.

## Experiment matrix

| Area | Population and status | Provenance |
|---|---|---|
| Historical synthesis | Phase 3–11 and 14 registry verified | `HISTORICAL_FROZEN` |
| Arabic embedding | 141 DEV queries; E5, BGE-M3, Arabic-Retrieval raw/light/aggressive; `RUN` | `PHASE15_DEV` |
| Dialect | 60 generated variants audited; 59 valid; 53 answerable paired metrics; `RUN` | `PHASE15_DEV` |
| Hard reranking | Corrected 46-query hard slice; `RUN` | `PHASE15_DEV` |
| Citation counterfactual | 40 persisted candidate answers; `RUN` | `PHASE15_DEV` |
| Abstention sensitivity | 141 score-only DEV records; `RUN` | `PHASE15_DEV` |
| Latency | 20-query fixed subset, batch 1, three warmups; `RUN` | `PHASE15_DEV` |
| Generator | Matched 80; fallback `RUN`, ALLaM blocked before scoring | `PHASE15_DEV` |

## Historical evidence: Phases 3–11 and 14

The Phase 3 split and Phase 4–6 artifacts establish the frozen evaluation
context. On the Phase 5 150-query challenge, `legal-structure-v1` achieved
Recall@10 0.7392, MRR@10 0.6916, and nDCG@10 0.6837, versus 0.7278, 0.6460,
and 0.6394 for `fixed-256-v1`. Its citation precision at rank 1 was 0.1299
versus 0.0363 for fixed-256. The Phase 5 decision selected
`legal-structure-v1`, while retaining an authoritative-statute revalidation
requirement.

Phase 7's frozen DEV dense baseline had Recall@10 0.0957 and nDCG@10 0.0671;
BM25 had Recall@10 0.2057 and nDCG@10 0.1385. Phase 8's selected RRF hybrid
had Recall@10 0.2021, MRR@10 0.1410, and nDCG@10 0.1481. These are aggregate
DEV results, not claims of statutory correctness.

Phase 9 recorded input gold-evidence coverage at 8 as 0.2695, complete gold
coverage as 0.2553, and zero unresolved sources in its tracked audit. Phase 10
Stage-D recorded supported-answer coverage 0.0922, supported-answer precision
0.1368, complete gold-evidence use 0.3158, false-answer rate 0.0526, and
unanswerable abstention recall 0.9474. Phase 11 deterministic extraction
artifacts remain candidate/normalization diagnostics and explicitly do not
provide an independent semantic-gold denominator.

## Arabic embedding experiment

The locked model is `omarelshehy/Arabic-Retrieval-v1.0` at revision
`899f6e1b765915a72d5e4ace6bb2b221715550d8`, with the tracked lock specifying
Apache-2.0, 768 dimensions, mean-token pooling, float32, CPU
sentence-transformers, empty query/passage prefixes, and L2 normalization after
encoding. All new embedding comparisons used CPU sentence-transformers.

On 141 DEV queries, BGE-M3 raw scored Recall@10 0.0957, MRR@10 0.0620, and
nDCG@10 0.0671. E5 raw scored 0.0922, 0.0307, and 0.0446. Arabic-Retrieval
scored 0.1418, 0.1029, and 0.1046 under raw normalization; 0.1241, 0.0998,
and 0.0971 under light; and 0.1206, 0.0920, and 0.0917 under aggressive.
CompleteEvidenceRecall@10 was 0.1489, 0.1348, and 0.1277 for Arabic raw,
light, and aggressive respectively. The raw-vs-BGE Recall@10 delta was +0.0461
with 95% CI [-0.0106, 0.1028]. The normalization deltas and all other metrics
are in `data/evaluation/phase15_embedding_metrics.json`.

These results are DEV-only and did not promote a model or normalization policy.

## Normalization and hybrid retrieval

The historical Phase 4 challenge reports identical aggregate raw, light, and
aggressive metrics and selected `arabic-raw-v1`; this is a bounded challenge
finding, not evidence that normalization is universally irrelevant. The new
Arabic-Retrieval comparison shows model-dependent normalization differences.

The historical hybrid result improves aggregate quality over dense-only, but the
tracked evidence does not support the stronger frozen wording “across Arabic
and English” with a language-stratified hybrid estimate. The research-question
status is therefore `INCONCLUSIVE`, not a production recommendation.

## Hard-query reranking

The corrected pre-registered hard rule is a deterministic OR over
`multi_evidence`, `exact_provision`, `authority`, `deadline`, `cross_language`,
`long_query`, or `min_pre_rerank_relevant_rank_for_hard=20`. It selected 46 DEV
queries; the earlier 19-query result is superseded for implementation mismatch.

On the corrected slice, hybrid-plus-reranker versus hybrid produced Recall@10
delta +0.0435, 95% CI [0.0000, 0.1087], with 2 wins, 44 ties, and 0 losses;
MRR@10 delta +0.0101, CI [0.0000, 0.0275], with 3 wins, 43 ties, and 0 losses.
The full four-metric result and rank-biserial values are in
`data/evaluation/phase15_reranking_metrics.json`. This is an operational
retrieval result on an enriched slice, not proof of legal-answer correctness.

## Dialect robustness

The content-only audit inspected all 60 existing texts without reading scores.
It found 59 valid variants—Egyptian 20, Gulf/Saudi 19, Levantine 20—and one
invalid Gulf/Saudi variant,
`dialect-gulf_saudi-3c9a92757a1c555d`, because it concatenated the MSA question
with a second paraphrase. The text was not regenerated. The valid-variant ID
hash is `5b7705c67a0a880cebba4084dfbe5f10b63bedf93ff97de192f5445bbf3d9cc2`.
The private per-variant audit is ignored; the tracked summary is
`data/evaluation/phase15_dialect_content_validity.json`.

Existing rankings were re-aggregated after this content exclusion without a
new retrieval scoring pass. Answerable paired counts are Egyptian 18,
Gulf/Saudi 17, Levantine 18, and pooled 53. Pooled dialect-minus-MSA
Recall@10 deltas were +0.0000 for BGE-M3, +0.0189 for BM25, +0.0189 for
hybrid, and +0.0377 for hybrid-plus-reranker. Pooled hybrid MRR@10 delta was
+0.0021; pooled BGE-M3 MRR@10 was -0.0031. Effects by dialect, system, metric,
and bootstrap interval are in `data/evaluation/phase15_dialect_metrics.json`.

These are AI-generated/AI-validated diagnostic perturbations, not human dialect
gold. The mixed small effects and unequal effective counts make the RQ status
`INCONCLUSIVE`.

## Citation verification counterfactual

The authoritative population is 40 schema-parsed candidate answers: 29
verifier-defective and 11 non-defective. The pre-defect-surface rate is 29/40
= 0.725; post-verification exposure is 0/40. The paired absolute risk reduction
is 0.725 with 95% CI [0.575, 0.850], from 29 discordant
before-positive/after-negative pairs. Defects are 18 semantic-support
rejections, 10 quotation/invalid-citation defects, and 1 other verification
failure. Coverage cost is reported by failure type in
`data/evaluation/phase15_citation_counterfactual.json`; no serving verifier was
weakened.

This measures contract-defect exposure only. It does not establish legal
correctness or prove that the verifier catches every substantive error.

## Abstention sensitivity

The uncalibrated score-only analysis retained coverage 1.000 with no gate,
0.893 at bottom-10 filtering, 0.745 at bottom-25 filtering, and 0.496 at
bottom-50 filtering. The corresponding tracked quality values and score-only
thresholds are in `data/evaluation/phase15_abstention_sensitivity.json`.
Relevance labels were not used to derive thresholds. Stage-D serving policy was
unchanged and no threshold was promoted.

## Latency

The controlled run used one fixed 20-query DEV subset, batch size 1, three
warmups, top-k 10, and CPU on arm64 macOS with Python 3.12.13 and the
sentence-transformers/numpy runtime class. p50/p95 milliseconds were:

| Operation | p50 | p95 | nDCG@10 |
|---|---:|---:|---:|
| keyword | 410.9 | 918.2 | 0.000 |
| BM25 | 77.1 | 583.7 | 0.056 |
| E5 | 77.4 | 584.0 | 0.056 |
| BGE-M3 | 115.2 | 623.3 | 0.056 |
| Arabic-Retrieval | 86.9 | 592.1 | 0.074 |
| hybrid | 722.1 | 757.0 | 0.056 |
| hybrid + reranker | 8925.2 | 11056.5 | 0.167 |

The full protocol and quality fields are in
`data/evaluation/phase15_latency_metrics.json`. Generator p50/p95 and disk
footprint are reported in `data/evaluation/phase15_generator_metrics.json`;
tokens/sec is omitted because it was not judged reliable.

## Generator matched-80 experiment

The frozen population is exactly 80 cases: 31 answerable with gold present, 30
answerable with gold absent, and 19 explicitly unanswerable. Query IDs and
context blocks were held constant. Extractive and Qwen3-4B outputs were reused
under the recorded historical fingerprints. The fallback lock is
`abdelrahman-alkhodary/qwen2.5-1.5b-arabic-instruct` revision
`06d27020b3ac3d9058b7eebded9754c8e10fa6bd`, Apache-2.0, with a 512-token
output limit. It ran locally with transformers on MPS, bfloat16, and a tracked
3,103,349,403-byte snapshot.

The fallback generated 80/80 invalid outputs under the frozen response
contract: invalid-generation rate 1.0, zero parsed answers, zero verified
answers, and zero supported coverage/precision. This is a negative DEV result,
not a model promotion decision. ALLaM remained
`BLOCKED_BEFORE_SCORING_NO_TRUSTWORTHY_4BIT_LOCAL_ARTIFACT`; it has no matched
results. The exact subset, metrics, and runtime audit are tracked in
`data/evaluation/phase15_generator_metrics.json` and
`data/evaluation/phase15_runtime_audit.json`.

## Automated diagnostic audit

The 30-case audit was selected before this final analysis and is enriched for
diagnostic coverage. It uses a fixed deterministic evidence-rule adjudication
followed by a second consistency-check pass; the second pass is not an
independent annotator. No human labels, expert labels, legal-expert review, or
human gold were produced.

| Outcome | Count |
|---|---:|
| Confirmed failure | 23 |
| Borderline/no confirmed failure | 7 |
| Uncertain | 0 |

Among confirmed failures, the 12-category taxonomy contains semantic retrieval
failure 8, lexical mismatch 6, reranker failure 2, and generator hallucination
1. Six additional confirmed cases are `INVALID_GENERATION_CONTRACT` and are
not mixed into that taxonomy. The corrected tracked aggregate is
`data/evaluation/phase15_error_analysis.json`; the private evidence-bearing
record is intentionally ignored at
`artifacts/private/phase15_evaluation/review/phase15_30_case_automated_adjudication.json`.

The initial-model versus rule-based-audit category comparison is limited to 15
cases with an available initial category and a final non-null 12-category
primary category: 10 agreements, rate 0.667. Disagreements are four
`semantic retrieval failure -> lexical mismatch` and one
`missing source -> semantic retrieval failure`. Contract failures, borderline
cases, null categories, and unavailable initial suggestions are excluded from
this denominator. This is not human-AI agreement.

The taxonomy is an operational diagnostic: it identifies the earliest
demonstrable failure mechanism in the persisted evidence. BM25 mismatch is a
lexical operational label; dense/hybrid misses identify the retrieval
subsystem but do not prove that upstream corpus or chunking causes are absent;
reranker failure is stronger only when relevant evidence is present before and
lost after reranking. No category proves causal attribution.

## Research questions

The seven frozen questions, statuses, populations, evidence, effects, and
limitations are recorded in `data/evaluation/phase15_research_questions.json`.

| RQ | Status |
|---|---|
| Structure-aware chunking versus fixed chunks | SUPPORTED |
| Effect of light Arabic normalization | PARTIALLY_SUPPORTED |
| Hybrid versus dense across Arabic and English | INCONCLUSIVE |
| Hard-query reranking | SUPPORTED |
| Dialectal robustness | INCONCLUSIVE |
| Citation verification | SUPPORTED for measured contract defects |
| Zero-cost local generation | PARTIALLY_SUPPORTED; the Phase 15 fallback was a negative contract result |

The statuses deliberately retain negative and inconclusive findings. They do
not authorize production tuning or model promotion.

## Figures and reproducibility

The sanitized, reproducible figures are generated from tracked aggregates by
`kawaneen.phase15.figures.build_report_figures` and stored under
`docs/reports/figures/phase15/`:

![Phase 5 chunking](figures/phase15/chunking-structure-aware-vs-fixed.svg)

![Retrieval quality versus latency](figures/phase15/retrieval-quality-vs-latency.svg)

![Dialect paired effect](figures/phase15/dialect-msa-paired-effect.svg)

![Generator outcome](figures/phase15/generator-outcome-support-coverage.svg)

![Automated diagnostic composition](figures/phase15/automated-diagnostic-composition.svg)

![Citation verifier](figures/phase15/citation-verifier-pre-post.svg)

Private query text, qrels, source snippets, dialect text, raw model output,
review packets, and diagnostic rationales remain ignored/private. No private
artifact is tracked.

## Production implications and limitations

Phase 15 makes no production change. It does not promote Arabic-Retrieval,
normalization, reranking, a score threshold, ALLaM, or the 1.5B fallback. The
fallback's failure is an implementation/contract result under one bounded
local configuration, not a general model-quality theorem.

Important limitations are:

- Phase 6 AI-reviewed records are not human gold or expert gold.
- The automated 30-case audit is not human or expert adjudication and is not a
  prevalence estimate; it is enriched and operational.
- Dialect variants are AI-generated/AI-validated diagnostics, with one invalid
  concatenation excluded before metric aggregation and no human dialect gold.
- New experiments are DEV-only; protected HOLDOUT data was not rerun or
  accessed.
- ALLaM was blocked before scoring, so no ALLaM quality claim is made.
- No qualified unattended statutory parser was established.
- Citation verification results address contract defects, not legal correctness.
- Latency is hardware/runtime-specific and must not be generalized across
  devices.
- No result establishes real-world legal safety or correctness.

## Finalization and handoff

`phase15 finalize` requires the frozen packet and audit hashes, the corrected
content audit, all required experiment artifacts in `RUN` or concrete
`BLOCKED` state, the text-free error analysis, all seven RQs, the final report,
the evidence registry, and zero protected-artifact violations. Human review
progress is not a completion requirement and remains empty; no human decisions
are fabricated.
