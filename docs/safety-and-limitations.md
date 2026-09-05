# Safety and limitations

This is the canonical public-facing safety document for Kawaneen. It describes
the concrete controls and boundaries a recruiter, evaluator, or demo viewer
should understand.

## Evidence-first behavior

Kawaneen retrieves source-linked evidence before it can answer. The serving
stack carries document/chunk identity, provenance, and exact quote material
through context assembly. Citation verification checks the returned reference
and support contract after generation. A citation failure is not silently
converted into a fluent answer.

## Deterministic answerability and abstention

Jurisdiction, evidence coverage, and answerability are evaluated before Stage-D
generation. Unsupported or insufficiently evidenced requests return an
intentional abstention. The public demo has an even narrower deterministic
gate: only positive lexical evidence can produce an exact synthetic evidence
response, otherwise it returns `INSUFFICIENT_DEMO_EVIDENCE`. Thresholds and
production policy were not changed during this closeout.

## Jurisdiction and unsupported queries

The full serving API is scoped to Saudi Arabia (`SA`) and should not be treated
as a general Arabic-law system. The public profile accepts only
`KAWANEEN_DEMO`, which is fictional and never presented as Saudi law. Queries
outside the configured scope, without sufficient evidence, or asking for a
legal conclusion the system cannot support must abstain or return a bounded
error. Synthetic passages cannot answer real-law questions.

## Generator and invalid-output handling

Stage-D is direct-only and bounded by a strict schema, input/output caps,
timeout, and zero automatic retries. Malformed generation, unsupported claims,
invalid references, or failed citation verification are rejected or abstained.
The Phase 15 fallback-generator result is retained honestly: all 80 matched
cases produced invalid outputs under the frozen contract. ALLaM was blocked
before scoring due to missing trustworthy local 4-bit provenance. Neither is a
hidden success or a general model-quality conclusion.

## Private corpus boundary

The real legal corpus, raw source files, per-example evaluation evidence,
private model caches, and review packets remain local and ignored. Source
registry permission is not a blanket redistribution or scraping license.
Fresh clones do not contain the full serving assets. Do not paste private legal
or customer data into public issues, screenshots, demos, or external tools.

## MLflow privacy boundary

MLflow is optional local observability. It records safe metadata such as
configuration identity, model/revision identifiers, stage names, statuses,
counts, and sanitized request metadata. It does not receive raw query text,
query hashes excluded by the privacy contract, legal evidence, generated
answers, extracted text, quote text, private paths, or stack traces containing
user content. The SQLite database and artifact store are intentionally ignored.
The tracked [portfolio export](mlflow-evidence.md) is text-only evidence, not
the database.

## Synthetic demo boundary

The public demo is not a smaller copy of production. It is a fictional,
retrieval-first profile with precomputed E5 vectors, exact NumPy search,
optional top-4 reranking, deterministic exact-passage responses, bounded
inputs, no LLM, and no private or Saudi source text. Local qualification means
only that the recorded constrained local run met its own gates; it is not a
Hugging Face host or production performance guarantee. No live public URL is
claimed.

## Extraction limitations

Deterministic extraction preserves source spans and has precision/safety
checks, but the repository does not establish complete semantic recall/F1
against human legal gold. The Phase 11 hybrid configuration remains
experimental and limited. Extracted candidates require human review before
any real legal or operational decision.

## Licensing and metadata limitations

Source categories carry different licences, terms, privacy risks, and access
conditions. Official-text status does not automatically permit bulk
acquisition, automated extraction, public quotation, or mirroring. The full
corpus metadata is incomplete for several structured filter fields, so a
working filter is not evidence of metadata relevance quality.

## Evaluation-label limitations

The frozen evaluation material includes externally AI-reviewed or
AI-adjudicated engineering labels, not independent human legal-expert gold.
Phase 15 diagnostic populations are selected for failure analysis and are not
population-prevalence samples. DEV-only deltas with confidence intervals that
include zero are reported as partial or inconclusive. No end-to-end legal
correctness accuracy metric exists.

## Latency and resources

The full local system loads BGE-M3, the BGE reranker, Qwen through Ollama,
Qdrant, and supporting services. It requires a private artifact root and
substantial memory/disk; one measured local run with approximately 7.75 GiB
OOM-killed the API, while the later approximately 11.66 GiB Docker engine
completed the frozen Compose verification. Phase 15 measured hybrid-plus-
reranker p50/p95 latency of 8,925.2/11,056.5 ms on a fixed 20-query CPU
subset. These are environment-specific observations, not universal SLAs.

## No legal advice or correctness guarantee

Kawaneen is a portfolio/research system. It does not guarantee that a result is
legally correct, current, complete, enforceable, or applicable to a person’s
facts. It is not legal advice and is not a replacement for a qualified lawyer.
Any real use requires authorized data, jurisdictional review, human judgment,
and independent verification against the governing authority.

## Public status

The repository has a qualified local synthetic-demo profile, but no live
deployment is claimed and no Hugging Face Space has been published. The only
remaining intentionally manual portfolio action is recording and validating
the real three-minute demo video; Hugging Face publication remains optional.
