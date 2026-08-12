# Phase 4 Arabic normalization experiment report

Date: 2026-08-11  
Status: complete for the frozen local Phase-4 scope  
Selected policy: `arabic-raw-v1`  
Selected policy hash: `1a9976c5a2fb6cad72f7622b508ac02485fbb9d25e8b2c6c7d2b41b65edae8da`

## Policies

All policies first apply NFC, remove U+FEFF BOM, and normalize retrieval whitespace
and line breaks to a single space with surrounding trim. The policy definitions and
versioned hashes are in `configs/normalization/policies.toml` and the sanitized
manifest.

| Policy | Additional transforms |
| --- | --- |
| `arabic-raw-v1` | None beyond NFC, BOM cleanup, and retrieval whitespace. |
| `arabic-light-v1` | Remove tatweel U+0640; remove the explicit Arabic diacritic ranges U+0610–U+061A, U+064B–U+065F, U+0670, U+06D6–U+06ED; fold `أ/إ/آ/ٱ → ا`. |
| `arabic-aggressive-v1` | Light plus `ى → ي`, experimental `ة → ه`, Arabic-Indic and Extended Arabic-Indic digits to ASCII digits, and only the explicit punctuation allowlist: Arabic comma/semicolon/question mark/decimal/group separators and en/em/minus variants to their ASCII equivalents. |

No policy uses broad NFKC, stemming, lemmatization, transliteration, or legal
abbreviation expansion. Abbreviation expansion remains a later query-expansion
concern. Aggressive punctuation handling is allowlist-only and does not delete
punctuation or concatenate legal identifiers.

Policy hashes:

| Policy | SHA-256 |
| --- | --- |
| `arabic-raw-v1` | `1a9976c5a2fb6cad72f7622b508ac02485fbb9d25e8b2c6c7d2b41b65edae8da` |
| `arabic-light-v1` | `78e262b833d1de1229d5f1c8618d91ba8d3d0916929386e3fbbdaed7352b5120` |
| `arabic-aggressive-v1` | `6f9d6e001f948b03b3541ee5df099968c80231f5455959341d8e290aa2f2c3bd` |

## Corpus and challenge

The full eligible population was 104,588 units. The frozen retrieval candidate set was
12,000 units: 6,000 ALARB and 6,000 ArabiCCR, balanced at 1,500 per included unit
type. This candidate policy was fixed before retrieval results and was not changed
after inspection.

The private, deterministic challenge contains 150 query-target items, with 15 items
per slice: unchanged controls, alef forms, diacritics, tatweel, digit variants,
`ى/ي`, `ة/ه`, punctuation/legal identifiers, combined variation, and deliberate
collision-risk cases. It has 24 multi-relevant qrel items for duplicate or parallel
passages. The same query-target set is used for every policy. The generated query and
qrel files, source text, per-example results, and private error/collision examples are
not tracked.

## Intrinsic diagnostics

Rates below are computed on the frozen 12,000-unit subset. “Distinct collision” is the
share of normalized forms that combine distinct source forms; “unit collision” is the
share of units in such groups.

| Policy | Character change | Token change | Vocabulary compression | Distinct collisions | Unit collisions | Collision groups | Identifier failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 15.0273% | 0.0525% | 0.0051% | 0 | 0 | 0 | 0 |
| light | 61.2303% | 57.7346% | 5.9163% | 0.0356% | 0.0750% | 4 | 0 |
| aggressive | 62.8585% | 59.3349% | 10.5877% | 0.0445% | 0.0917% | 5 | 0 |

All three policies passed deterministic, idempotent, exact-display-preservation, and
identifier-safety gates. The most frequent aggregate transforms were whitespace
mapping (raw 2,138,183; light/aggressive 2,303,124), alef folding (461,148),
diacritic removal (108,154), tatweel removal (26,285), Arabic digit folding (213,295),
allowlisted punctuation folding (189,057), ya-maqsura folding (243,455), and
experimental ta-marbuta folding (584,573) where applicable. These are aggregate
counts only; no source text is included.

Synthetic safety fixtures cover Arabic and Western digits, article/date/decree and
reference-shaped identifiers, mixed digit forms, separator deletion, token
concatenation, empty/format-control input, and mixed Arabic/English text. For example,
`المادة (١٢/١٤٤٥) م/١٢٣` and its ASCII-digit equivalent preserve identifier structure,
while `قرار ١٢٣ / ٢٠٢٤ → قرار123/2024` is rejected. Synthetic collision fixtures
cover alef, ya-maqsura, and ta-marbuta equivalences. No real corpus snippets are
reported here.

## Controlled lexical retrieval ablation

The experiment used the same Unicode-word-or-single-punctuation tokenizer, BM25
`k1=1.2`, `b=0.75`, candidate set, seed `20260811`, and qrels for each symmetric
query/corpus pairing.

| Policy | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `arabic-raw-v1` | 0.872962 | 0.944772 | 0.965908 | 0.971556 | 0.976551 |
| `arabic-light-v1` | 0.872962 | 0.944772 | 0.965908 | 0.971556 | 0.976551 |
| `arabic-aggressive-v1` | 0.872962 | 0.944772 | 0.965908 | 0.971556 | 0.976551 |

Every pairwise comparison was 0 wins, 150 ties, 0 losses. The collision-risk slice
was identical for all policies: Recall@1 0.252516, Recall@5 0.709245, Recall@10
0.822193, MRR@10 0.855556, and nDCG@10 0.870245. The unchanged-control slice was
1.0 on every reported metric for every policy. Alef, diacritic, tatweel, digit, ya,
ta, punctuation, and combined-variation slice metrics were also identical across
policies; detailed sanitized slice values are in
`data/evaluation/phase4_normalization_metrics.json`.

The deterministic paired bootstrap used 2,000 replicates with seed `20260811`.
For raw versus light, raw versus aggressive, and light versus aggressive, the key
Recall@10, MRR@10, and nDCG@10 deltas were all 0.0 with confidence intervals
`[0.0, 0.0]`.

## Selection and rejected transformations

The decision rule first eliminates hard-gate failures, then prefers the least
destructive policy when retrieval is effectively tied. All three candidates passed
the hard gates, but all retrieval metrics, every paired delta, and every important
slice comparison tied exactly. `arabic-light-v1` therefore did not meet the
predefined meaningful-improvement requirement, and `arabic-aggressive-v1` did not
clearly improve over light. `arabic-raw-v1` was selected because it preserves the most
distinctions while retaining retrieval-whitespace cleanup and producing zero observed
distinct-form or unit collisions in this scope.

Explicitly rejected from these policies are broad NFKC, stemming, lemmatization,
transliteration, legal-abbreviation expansion, unrestricted punctuation-category
rewrites, punctuation deletion, and any transform that can concatenate legal
identifiers. `ة → ه`, `ى → ي`, digit folding, and punctuation folding remain tested
aggressive experiments, not selected product behavior.

## Derived view and manifests

The selected private derived view contains policy-versioned normalized records with
canonical/unit IDs, exact `display_text`, derived `search_text`, policy ID/hash,
source/search hashes, and provenance. All three private variants were materialized
during the experiment; the selected raw view was materialized after selection. Private
Parquet, queries, qrels, per-example results, temporary indexes, and source-text error
reports remain ignored under `artifacts/private/phase4_normalization/`.

Tracked sanitized outputs are:

- `data/manifests/normalization/policies.json` — policy definitions and hashes;
- `data/manifests/normalization/phase4_manifest.json` — frozen scope, provenance hashes,
  gates, diagnostics, and selection;
- `data/evaluation/phase4_normalization_metrics.json` — aggregate metrics, slice
  values, pairwise outcomes, and confidence intervals.

## Phase 4 gate table

| Gate | Result | Evidence |
| --- | --- | --- |
| policy definitions and version/hash identity | passed | three pure policies; TOML/code mirror test |
| Phase 3 immutability and display preservation | passed | 12,000 records per policy; canonical hashes unchanged |
| frozen candidate policy | passed | 12,000 fixed IDs; 8 included content types; frozen before results |
| independent challenge construction | passed | 150 items, 10 balanced slices, 24 multi-relevant qrel items |
| determinism/idempotency/identifier safety | passed | zero failures for all policies |
| collision constraints | passed | maximum distinct-form rate 0.0445%; maximum unit rate 0.0917% |
| controlled lexical ablation | passed | identical corpus/tokenizer/BM25/seed/qrels across policies |
| selection and private derived view | passed | raw selected; manifests and ignored view produced |
| privacy and governance | passed | no real-data challenge artifacts tracked; Phase 2/3 permissions unchanged |
| full verification | pending final handoff run | focused/full tests, coverage, Ruff, Pyright, pre-commit, `make check`, and hash/diff checks |

Phase 4 is ready to close after the final verification commands pass. Phase 7 must
revalidate `arabic-raw-v1` against the later human evaluation set before treating this
selection as a production retrieval decision.
