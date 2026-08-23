# Phase 10 final DEV selection

Status: frozen for Phase-10 DEV reproducibility. The selected configuration is
`qwen-ollama-stage-d`. No holdout result was accessed in Phase 10.

## Final architecture

The selected path is:

1. frozen Phase-8 DEV top-8 selection;
2. Phase-9 authoritative provenance resolution and canonical ContextPack assembly;
3. real Qwen tokenizer budgeting with no mid-unit truncation;
4. request-local QuoteRegistry identifiers (`Q001`, `Q002`, ...);
5. deterministic Stage-D AnswerabilityPolicy before generation;
6. Qwen direct-only JSON generation;
7. server-side QuoteRegistry resolution into Phase-9 CitationRequests;
8. unchanged Phase-9 citation verification and deterministic rendering.

The model does not emit quotation text or source metadata. Final direct text is
constructed from verified authoritative registry entries. Extractive generation
remains benchmark-only and is not an automatic fallback.

## Frozen configuration

The text-free selection manifest is
`data/manifests/generation/phase10_selected_configuration.json`.

- model: `Qwen/Qwen3-4B-Instruct-2507`
- immutable HF/model revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Ollama tag: `qwen3:4b-instruct-2507-q4_K_M`
- Ollama digest: `sha256:0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`
- tokenizer: `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`, local-only loading required
- AnswerabilityPolicy: `phase10-stage-d-answerability-policy-v1`, hash
  `8c990acd4d983da61baf3b8d72a4150f6f9f75b30dbf0b57ef18660fc84930e0`
- direct-only schema hash:
  `9aa4009056bc2a58ea079c1be1e0a57537044bbb354e167416e816132b26f9de`
- prompt/template hash:
  `316e0496cff5e7fbd8e86eee11cba07f0bd9b193e04bead5dc7864aae36b357e`
- QuoteRegistry policy: `phase10-stage-d-quote-registry-v1`
- verifier: authoritative Phase-9 citation verifier
- timeout/retries: 60 seconds / 0
- input/output caps: 3584 / 512
- temperature: 0
- sampling: disabled

## Engineering history

| Configuration | Disposition | Engineering result |
|---|---|---|
| Extractive | Benchmark-only | Deterministic safety baseline; not a fallback |
| Stage A | Historical experiment | Excessively conservative and malformed-output prone |
| Stage B | Historical experiment | Provider JSON Schema and direct-claim contract correction |
| Timeout diagnostics | Historical diagnostics | Runtime and truncation root-cause instrumentation |
| Stage C | Historical experiment | Compact QuoteRegistry references and 60-second timeout |
| Stage D | Selected | Source-eligibility gate plus direct-only generation |

Stage-A/B/C and diagnostic artifacts remain preserved as historical evidence.
They are not selected configurations and are not overwritten or deleted.

## Stage-D DEV results

The frozen DEV run contains 160 valid completed records: 131 generation results
and 29 deterministic pre-generation policy abstentions. Final outcomes are 95
answers and 65 abstentions.

| Metric | Result |
|---|---:|
| SupportedAnswerPrecision | 13/95 = 0.136842 |
| SupportedAnswerCoverage | 13/141 = 0.092199 |
| ContextInsufficientAbstentionRecall | 38/103 = 0.368932 |
| UnanswerableAbstentionRecall | 18/19 = 0.947368 |
| FalseAnswerRate | 1/19 = 0.052632 |
| FalseAbstentionRate | 9/38 = 0.236842 |
| ValidCitationRate | 97/97 = 1.000000 |
| ClaimCitationCoverage | 95/95 = 1.000000 |
| GoldCitationHitRate | 13/97 = 0.134021 |
| CompleteGoldEvidenceUse | 12/38 = 0.315789 |
| Invalid-generation rate | 0/160 = 0.000000 |
| Final-answer coverage | 95/160 = 0.593750 |

Structural safety invariants all remain zero: fabricated references or metadata,
altered quotations, model quotation exposure, uncited claims, unsupported
interpretations, jurisdiction/advice/currentness violations, and invalid output
reaching final rendering.

## Qualitative label review

The nominal automated value remains `FalseAnswerRate = 1/19`. Manual review of
the sole final answer in the explicitly-unanswerable population found that the
Phase-6 label is ambiguous: the retrieved precedent is materially responsive
to a reasonable interpretation, although the label remains unchanged.

This item is recorded only as the aggregate candidate
`ANSWERABILITY_LABEL_REVIEW_CANDIDATE`. No Stage-D rule was added or tuned for
this isolated case, and no qrel, metric, or population membership was changed.

The final disposition is:

`QWEN_STAGE_D_SELECTED_AFTER_QUALITATIVE_REVIEW`

The selection reflects passed structural grounding gates, zero invalid
generation, zero transport failures, 100% citation integrity, and safer
abstention behavior. It does not claim exhaustive semantic legal correctness.

## Limitations and boundary

- This is DEV evaluation only.
- Phase-6 evaluation is independent-AI-reviewed, not human-gold.
- Qrels are non-exhaustive.
- No synchronized exhaustive legal semantic evaluation exists.
- No semantic NLI verifier is present.
- Direct quotation does not inherently establish universal legal applicability.
- Answerability policies are intentionally conservative.
- No holdout result has been inspected in Phase 10.

The selected Stage-D manifest is frozen before any future holdout access. Any
future holdout evaluation must use this configuration unchanged.
