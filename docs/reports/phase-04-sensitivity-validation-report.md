# Phase 4 final sensitivity validation report

Date: 2026-08-12  
Scope: bounded validation only; no Phase 2/3 data, primary challenge, or primary qrels changed.

## Frozen primary experiment

The existing 150-query challenge was loaded read-only as
`phase4-primary-challenge-v1`. The candidate set remained the existing 12,000 units,
with the existing candidate IDs, qrels, tokenizer, BM25 parameters, seed, and raw,
light, and aggressive policies. The primary challenge file hashes were unchanged
before and after validation.

The tokenizer is `unicode-word-or-single-punctuation-v1` and has no hidden Arabic
normalization: synthetic checks distinguish alef forms, Arabic and Western digits,
ya/maqsura, ta-marbuta/ha, and Arabic/ASCII punctuation. Probe construction invokes
no candidate normalizer; perturbations are defined independently.

## Why the original 150 queries tied

The primary challenge was dominated by long canonical passages rather than short
orthography-sensitive queries. Median source/query lengths ranged from 139 to 468
tokens by slice. The controlled changes frequently changed query/corpus overlap and
BM25 scores, but the relevant target was already ranked first or the remaining lexical
context was redundant enough that the top-10 result did not change.

The audit recorded 143/150 aggressive and 128/150 light queries with target-overlap
changes relative to raw, while all three aggregate retrieval policies remained exactly
tied at the primary level. The primary ablation remains:

| Policy | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.872962 | 0.944772 | 0.965908 | 0.971556 | 0.976551 |
| light | 0.872962 | 0.944772 | 0.965908 | 0.971556 | 0.976551 |
| aggressive | 0.872962 | 0.944772 | 0.965908 | 0.971556 | 0.976551 |

All primary pairwise comparisons were 0 wins, 150 ties, 0 losses; key paired CIs were
exactly `[0.0, 0.0]`. The primary result is therefore a genuine ranking tie despite
non-zero lexical/score effects, not evidence that the normalizer is inert.

## Primary sensitivity by slice

Each primary slice contains 15 queries. Percentages below report raw mismatch
preservation, intended-transform overlap repair, non-zero target-score change, and
rank/top-10 change. The tracked JSON also records policy-specific query, overlap,
score, and top-10 rates.

| Slice | Median length | Raw mismatch preserved | Intended repair | Non-zero light score delta | Light rank/top-10 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| unchanged control | 335 | 0.0% | 0.0% | 100.0%* | 86.7% |
| alef forms | 249 | 100.0% | 33.3% | 100.0% | 100.0% |
| diacritics | 339 | 100.0% | 20.0% | 100.0% | 93.3% |
| tatweel | 253 | 100.0% | 26.7% | 100.0% | 93.3% |
| digit variants | 312 | 100.0% | 13.3% | 100.0% | 93.3% |
| ya/maqsura | 186 | 100.0% | 20.0% | 100.0% | 93.3% |
| ta-marbuta/ha | 139 | 100.0% | 40.0% | 100.0% | 66.7% |
| punctuation/identifiers | 468 | 100.0% | 20.0% | 100.0% | 86.7% |
| combined variation | 372 | 100.0% | 26.7% | 100.0% | 80.0% |
| collision risk | 156 | 100.0% | 33.3% | 100.0% | 40.0% |

\* Controls contain no deliberate query perturbation. Their target scores and top-10
IDs can still differ because the corpus index is normalized symmetrically; this is
distinguished from an intended query perturbation effect in the private audit.

The primary classification counts were: 15 unchanged controls, 4 score-change/no-
rank-change cases, 31 rank/top-10 changes, and 100 redundant or non-repairing cases
where the intended transform did not improve the anchor target overlap.

## Separate 60-item sensitivity probe

The probe is `phase4-sensitivity-probe-v1`, seed `20260812`, with 10 slices × 6
items: controls, alef forms, diacritics, tatweel, digit variants, ya/maqsura,
ta-marbuta/ha, punctuation/identifiers, combined variation, and collision risk.
Queries use deterministic short windows from the frozen canonical display text, with
median eight lexical tokens and a 2–8 token construction bound. Perturbations and
qrels are policy-independent. The probe contains 51 single-target items and 9
multi-relevant items, including duplicate/parallel canonical units; all text-bearing
probe files remain private.

| Slice | Raw mismatch preserved | Intended repair | Non-zero light score delta | Light rank/top-10 change |
| --- | ---: | ---: | ---: | ---: |
| unchanged control | 0.0% | 0.0% | 100.0%* | 83.3% |
| alef forms | 100.0% | 83.3% | 100.0% | 100.0% |
| diacritics | 100.0% | 33.3% | 100.0% | 100.0% |
| tatweel | 100.0% | 66.7% | 100.0% | 83.3% |
| digit variants | 100.0% | 100.0% | 100.0% | 83.3% |
| ya/maqsura | 100.0% | 33.3% | 100.0% | 83.3% |
| ta-marbuta/ha | 100.0% | 50.0% | 100.0% | 83.3% |
| punctuation/identifiers | 100.0% | 16.7% | 100.0% | 83.3% |
| combined variation | 100.0% | 100.0% | 100.0% | 100.0% |
| collision risk | 100.0% | 100.0% | 100.0% | 50.0% |

The probe harness clearly detects normalization effects. Raw scored 0.575556 MRR@10,
0.733333 Recall@10, and 0.618291 nDCG@10. Light scored 0.680714 MRR@10, 0.766667
Recall@10, and 0.705032 nDCG@10. Aggressive scored 0.642917 MRR@10, 0.750000
Recall@10, and 0.671400 nDCG@10.

The raw-to-light paired deltas (light minus raw) were +0.105159 MRR@10, +0.033333
Recall@10, and +0.086740 nDCG@10. Converting the stored left-minus-right intervals
to light-minus-raw, the deterministic 2,000-replicate intervals were respectively
`[0.051984, 0.166270]`, `[0.0, 0.083333]`, and `[0.041486, 0.139337]`.
The light-versus-aggressive comparison did not clearly favor aggressive: light minus
aggressive was +0.037798 MRR@10, +0.016667 Recall@10, and +0.033631 nDCG@10, with
intervals crossing or touching zero for the key metrics.

Controls contain no deliberate perturbation: their construction query is unchanged,
and their intended-repair rate is zero. Because the controls are evaluated against
symmetrically normalized indexes, normalization can still change target scores and
ranking; those changes are corpus-index effects, not evidence that challenge
construction perturbed the control query.

Selected probe slice retrieval results (Recall@10 / MRR@10) were:

| Slice | Raw | Light | Aggressive |
| --- | ---: | ---: | ---: |
| unchanged control | 0.833 / 0.750 | 0.833 / 0.833 | 0.833 / 0.688 |
| alef forms | 0.667 / 0.472 | 0.833 / 0.722 | 0.833 / 0.722 |
| diacritics | 0.833 / 0.667 | 0.833 / 0.833 | 0.833 / 0.688 |
| tatweel | 0.833 / 0.583 | 0.833 / 0.833 | 0.833 / 0.688 |
| digit variants | 0.667 / 0.667 | 0.667 / 0.667 | 0.667 / 0.667 |
| ya/maqsura | 0.667 / 0.583 | 0.667 / 0.583 | 0.667 / 0.667 |
| ta-marbuta/ha | 0.667 / 0.667 | 0.667 / 0.667 | 0.667 / 0.667 |
| punctuation/identifiers | 0.833 / 0.556 | 0.833 / 0.639 | 0.667 / 0.583 |
| combined variation | 0.833 / 0.556 | 1.000 / 0.774 | 1.000 / 0.806 |
| collision risk | 0.500 / 0.256 | 0.500 / 0.256 | 0.500 / 0.256 |

The complete sanitized slice metrics, score/overlap rates, pairwise outcomes, and all
confidence intervals are in the sanitized
`data/evaluation/phase4_sensitivity_metrics.json`.

## Decision

The primary experiment’s frozen selection remains `arabic-raw-v1`; its 150-query
challenge was not regenerated or replaced. Under the user-defined final sensitivity
rule, the short-query probe provides meaningful light-normalization gains without a
material control/collision regression, while aggressive does not clearly beat light
and has a control regression. The bounded validation therefore recommends
`arabic-light-v1` for downstream Phase-7 revalidation, not unconditional production
promotion.

This result is a sensitivity recommendation, not a Phase 5 start and not the Phase 6
human evaluation set. Phase 7 must revalidate light against the later human evaluation
set and the existing safety/collision gates before any production retrieval decision.

## Verification and privacy

Tracked sensitivity outputs contain only hashes, IDs/counts, rates, metrics, policy
names, and configuration metadata. Per-query audit rows, probe queries/qrels, private
results, and all derived text remain ignored under
`artifacts/private/phase4_normalization/sensitivity_validation/`.
