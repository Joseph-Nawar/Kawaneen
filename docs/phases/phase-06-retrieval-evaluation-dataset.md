# Phase 6 — Frozen Retrieval Evaluation Dataset

Phase 6 creates a private, review-gated retrieval evaluation set for later
retrieval work. It does not implement retrieval optimization, embeddings,
dense or hybrid search, reranking, RAG, APIs, or UI.

## Corpus scope

The corpus is the complete eligible governed ALARB and ArabiCCR Phase 3
canonical corpus, frozen before candidate generation. ALARB includes `facts`,
`court_reasoning`, `applicable_laws`, and `verdict`. ArabiCCR includes
`EVENTS`, `REASONING`, and `RULING`; `case_text` is permitted only when no
structured section exists. OCR-derived material and the MOJ-derived statute
seed are excluded from retrieval gold. Canonical file hashes, source revisions,
document/unit counts, IDs, and the content-policy hash are recorded in the
text-free corpus manifest.

The available benchmark query/relevance source is unavailable in this
checkout. Phase 6 therefore uses document-derived candidates and does not
claim benchmark-derived instances. The MOJ seed is not authoritative current
statute text; explicit provision questions can refer only to references found
inside governed judgments.

## Schema and evidence

Each private item has a stable `query_id` and `intent_id`, optional variant
identity, query text, language/register, query type, jurisdiction and temporal
scope, creation method, answerability/reason, structural difficulty, source
document IDs, canonical evidence groups, optional authoritative article IDs,
the selected `legal-structure-v1` policy hash, concise answer, Phase 5 citation
anchors, review metadata, dataset version, and split.

Gold relevance is defined over exact canonical source spans: grade 2 is directly
sufficient or required, grade 1 is supporting, and grade 0 is irrelevant.
Multiple evidence groups are supported. Chunk qrels are deterministic derived
views of those spans and are never the source of truth.

## Candidate and review targets

The v3 private draft builds 370 evidence-qualified answerable candidates plus
25 externally accepted base unanswerables, selects 200 base intents in the
requested category allocation, and adds 40 intent-preserving variants: ten
simple Arabic, ten Egyptian Arabic, ten English, and ten Arabic-English
code-switched. All generated records start `human_verified=false`.

Before review, the source-balance audit compares category-level eligible
evidence opportunities with generated and selected records. Candidate caps use
measured opportunity share and deterministic structural ordering; they do not
use retrieval scores or force a 50/50 source split. The v3 semantic-target
regeneration produces 97 ALARB / 103 ArabiCCR base records (84 / 91 among
answerable base intents) without forcing a 50/50 split. Semantic opportunity counts are recorded by
category and source, including ArabiCCR `EVENTS`, `REASONING`, and `RULING`
parity diagnostics. The text-free audit is written to
`data/evaluation/phase6_v3_source_balance_audit.json`.

Evidence discovery rejects punctuation-only spans, structural list prefixes,
category mismatches, and facts-only holding/provision evidence. Query
generation is independent of retrieval rankings. A typed semantic proposition is
extracted from each canonical evidence span before both query and answer
generation; category validation and answer entailment validate that proposition
against the span. Direct query-to-answer and query-to-evidence leakage is a
hard gate while ordinary lexical overlap remains a review signal. v3 diagnostics
are private at
`artifacts/private/phase6_evaluation/draft-v3/review/review_diagnostics.jsonl`.
Lossless-for-source-text canonical handoff shards are private under
`artifacts/private/phase6_evaluation/draft-v3/handoff/canonical_review_shards/` with a
hash manifest and smaller source-context file.

Review packets show the question, answer, source excerpts with evidence
highlights, citations, and editable review fields. The allowed state path is
`draft → primary_reviewed → secondary_reviewed → adjudicated → frozen`.
Freeze requires primary review of every item, independent review of every
holdout item, independent recheck of at least 25% of dev, double review of all
unanswerable/hard/multi-evidence items, and no unresolved disagreements.

## Leakage-safe split

The provisional split is approximately 160 dev / 80 holdout. Connected
source-document groups and intent families are assigned together, so variants,
multi-document evidence, and source-linked records cannot cross splits. Twenty
dev records are marked `smoke`. Loaders default to dev and require explicit
`allow_holdout=True` to expose holdout.

## Privacy and freeze policy

Private text-bearing outputs are ignored under
`artifacts/private/phase6_evaluation/`. Tracked outputs contain no query text,
answers, evidence, qrels, excerpts, or review material. Privacy, duplicate,
near-duplicate, span, evidence-group, answerability, corpus-hash, and chunk
mapping checks run before scoring. Retrieval rankings are never used to select,
reject, rewrite, relabel, or split records. This remediation is versioned as
`phase6-retrieval-eval-draft-v3`; any later semantic edit must create a new
dataset version rather than mutate a frozen release.

The bounded external-review application is versioned as
`phase6-retrieval-eval-draft-v4`. It preserves accepted unanswerable records,
keeps canonical spans for corrected records, replaces only adjudicated base
records, and regenerates variants only after base semantic validation. v4 is a
private final-external-review candidate, not a frozen dataset.

The final v4 adjudication is applied as
`phase6-retrieval-eval-draft-v5`. It preserves the 25 accepted unanswerables,
applies only the declared corrections and replacements, rebuilds the 25
multi-evidence records with two necessary grade-2 groups, and regenerates all
40 robustness variants last. v5 is sent for changed-record-only external
source review; it is not frozen and contains no human-verified records.

## Externally AI-reviewed engineering release

`phase6-retrieval-eval-final-candidate-v1` is treated as the completed Phase 6
semantic/content construction release after its literal external AI source
review and adjudication. It is materialized privately as
`phase6-retrieval-eval-ai-reviewed-v1` without changing any record text,
canonical evidence, qrels, splits, or existing hashes.

This release is explicitly `externally_ai_reviewed` with review provenance
`independent_ai_source_review` and `human_verified=false`. It is not a
human-gold, expert-reviewed, or independently human-annotated benchmark. The
set underwent source-grounded generation, deterministic validation, external
AI source review and adjudication, but not independent human legal-expert
annotation.

The human review workflow remains available as an optional future upgrade path
for a publication-grade human-reviewed-v1. It does not block the current
engineering project or Phase 7 from using the AI-reviewed dev split and the
explicitly guarded holdout path. The text-free release manifest and report are
`data/manifests/evaluation/phase6_ai_reviewed_v1_manifest.json` and
`data/evaluation/phase6_retrieval_eval_ai_reviewed_v1_report.json`.

## Commands

```text
kawaneen evaluation plan
kawaneen evaluation build-draft
kawaneen evaluation build-draft-v3 --review-file /path/to/phase6_independent_ai_source_review_v2.jsonl
kawaneen evaluation build-draft-v4 --review-file /path/to/phase6_v3_external_ai_review_adjudication.jsonl
kawaneen evaluation build-draft-v5 --review-file /path/to/phase6_v4_final_external_ai_adjudication.jsonl
kawaneen evaluation balance-audit
kawaneen evaluation export-review
kawaneen evaluation import-review --file artifacts/private/phase6_evaluation/review/review_packet.jsonl
kawaneen evaluation validate
kawaneen evaluation stats
kawaneen evaluation freeze
kawaneen evaluation freeze-ai-reviewed
```

`freeze` remains the optional publication-grade human-review path and remains
blocked until its human review gates are satisfied. The engineering release
is materialized with `freeze-ai-reviewed`, which does not create human
attestations. Phase 7 must evaluate dev normally and access holdout only
through an explicit, audited `allow_holdout=True` path using the frozen
AI-reviewed release and its hash manifest.
