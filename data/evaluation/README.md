# Evaluation artifacts

Phase-specific tracked evaluation files contain aggregate metrics and text-free
metadata only. Real query text, qrels, source spans, snippets, and per-example
results remain private and ignored under the relevant phase artifact root.

Phase 6 uses `artifacts/private/phase6_evaluation/` for the full canonical
snapshot, candidate JSONL, canonical-span evidence, derived
`legal-structure-v1` qrels, source excerpts, and review packets. Tracked Phase
6 manifests contain only source revisions, canonical/policy/corpus hashes,
counts, distributions, diagnostics, and review status. The engineering
release is frozen as an externally AI-reviewed set, not as a human-verified
gold set.

The current bounded remediation is `phase6-retrieval-eval-draft-v2`: a
360-record evidence-qualified candidate pool, 200 selected base intents, and
40 intent-preserving variants. Benchmark query/relevance instances are
unavailable and no records are claimed as benchmark-derived. The source-balance
audit uses semantic opportunities and does not use retrieval rankings.

The current engineering release is
`phase6-retrieval-eval-ai-reviewed-v1`. It contains 200 base intents and 40
variants, preserves `human_verified=false`, and records
`independent_ai_source_review` as its provenance. It underwent source-grounded
generation, deterministic validation, external AI source review and
adjudication, but not independent human legal-expert annotation. It must not
be described as human-gold, expert-reviewed, or independently human-annotated.

Human review remains an optional future upgrade path for a publication-grade
human-reviewed-v1 and does not block the engineering Phase 7 workflow. The
text-free release manifest is
`data/manifests/evaluation/phase6_ai_reviewed_v1_manifest.json`; the report is
`data/evaluation/phase6_retrieval_eval_ai_reviewed_v1_report.json`.

The bounded pre-review source-balance report is
`phase6_source_balance_audit.json`. It contains only opportunity counts,
source/category distributions, sanitized rejection reasons, section-parity
diagnostics, and hashes; it contains no private text.
