# Kawaneen system model card

This is a system/component card. Kawaneen did not train BGE-M3, the BGE
reranker, Qwen, E5-small, or the other listed third-party models end-to-end.
The project contribution is the orchestration, contracts, deterministic rules,
evaluation design, serving boundary, privacy boundary, and reproducibility
record around frozen pretrained components.

## System summary

Kawaneen is a jurisdiction-aware Arabic legal-document intelligence system.
Its full local profile ingests a private/local corpus, retrieves candidate
evidence with BM25 and dense search, fuses and reranks it, assembles grounded
context, applies deterministic answerability and citation checks, and exposes
search, answers, extraction, document reads, readiness, and model metadata over
FastAPI with a Streamlit interface.

The public profile is a separate synthetic retrieval-first system. It uses the
fictional `KAWANEEN_DEMO` corpus and returns exact synthetic evidence passages;
it has no generative model.

## Intended use

- Inspectable Arabic legal/research retrieval and evidence workflows.
- Engineering evaluation of hybrid retrieval, reranking, grounding, citation
  contracts, structured extraction, and abstention.
- Local research against authorized private assets, or public walkthroughs
  against the synthetic demo only.

## Out-of-scope use

Kawaneen is not a legal decision-maker, legal advice service, production
compliance authority, or substitute for a qualified lawyer. Do not use it to
make an unsupervised legal determination, infer law outside the configured
jurisdiction, or treat synthetic demo passages as statutes.

## Component inventory and frozen identities

| Component | Identity/configuration | Role |
| --- | --- | --- |
| Sparse retrieval | BM25; sparse top-k 50 | Lexical candidate retrieval |
| Full-profile dense retrieval | `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181` | Dense candidate retrieval |
| Fusion | RRF `k=60`; sparse weight 1.0; dense weight 0.25; candidate count 20 | Candidate merge |
| Full-profile reranker | `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; raw-logit-v1; serving depth 8 | Candidate ordering |
| Stage-D generator | `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`; Ollama tag `qwen3:4b-instruct-2507-q4_K_M`; digest `sha256:0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` | Direct-only grounded generation |
| Answerability | `phase10-stage-d-answerability-policy-v1`; hash `8c990acd4d983da61baf3b8d72a4150f6f9f75b30dbf0b57ef18660fc84930e0` | Pre-generation support/abstention gate |
| Prompt contract | `phase10-stage-d-prompt-template-v1`; version hash `4e5fccb56222c0310136e4a3446ccfffb2af412bc2b6c73f6262fae4320c180c` | Bounded generation input |
| Citation verifier | Authoritative Phase 9 verifier; direct quote/provenance contract | Fail-closed citation support |
| Extraction | Phase 11 B2 selected as `PHASE11_HYBRID_EXPERIMENTAL_LIMITED`; deterministic candidate path is precision-audited, not recall/F1-complete | Structured spans and exports |
| Public dense retrieval | `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`; precomputed corpus vectors | Synthetic demo only |
| Public reranker | Same BGE reranker revision, top-4 only when locally qualified | Optional synthetic-demo ordering |

The full identity and source configuration hashes are tracked in
[`data/manifests/observability/phase16_serving_identity.json`](../data/manifests/observability/phase16_serving_identity.json)
and the frozen retrieval/generation manifests.

## Retrieval stack

The full profile runs BM25 and exact dense retrieval over the frozen corpus,
combines the first-stage rankings with the frozen weighted RRF configuration,
then reranks the fused candidate set. The Phase 8 selection and model lock are
immutable tracked inputs. Phase 17 separately verified exact NumPy/Qdrant
parity on 20 DEV queries at top-k 50, with zero ordering mismatches and maximum
score error `1.43e-7`.

## Reranking

The BGE reranker consumes the frozen exact chunk display text, uses CPU
sentence-transformers configuration with maximum length 1024 and raw logits,
and returns at most eight serving results from a 20-candidate fused set. The
Phase 15 hard-query slice showed a small directional effect, but its confidence
interval includes zero and it is not a population-wide correctness claim.

## Stage-D generation

Stage-D is direct-only, deterministic at temperature 0, with no automatic
retries, a 60-second timeout, input cap 3,584, and output cap 512. It receives
bounded grounded context and must satisfy a strict answer/quote-reference
schema before downstream citation verification. Qwen is a frozen pretrained
component; Kawaneen owns the policy, prompt, parsing, provenance, and
verification contracts.

The Phase 15 fallback experiment is intentionally recorded as a failure: the
locked `abdelrahman-alkhodary/qwen2.5-1.5b-arabic-instruct` revision
`06d27020b3ac3d9058b7eebded9754c8e10fa6bd` produced invalid outputs in all 80
matched cases. ALLaM was blocked before scoring because a trustworthy local
4-bit artifact was unavailable. Neither result is hidden or promoted.

## Citation verification

The verifier checks citation identity, quote/reference shape, source
provenance, and semantic support according to the Phase 9 contract. In the
Phase 15 persisted counterfactual, it reduced measured contract-defect
exposure from 29/40 to 0/40. This is a contract-defect result, not proof that
all substantive legal errors are caught.

## Extraction and abstention

Deterministic extraction proposes bounded candidates and preserves source
spans; the Phase 11 evidence supports precision/safety invariants, not an
independent semantic-gold recall or F1 denominator. The answerability policy
can abstain before generation when jurisdiction, evidence, or support gates
fail. Invalid generation and citation failures are fail-closed outcomes.

## Evaluation evidence

- [Phase 15 report](reports/phase-15-evaluation-and-experiment-report.md) —
  pre-registered DEV comparisons, generator failure, error analysis, and
  limitations.
- [Phase 16 report](reports/phase-16-observability-and-reproducibility.md) —
  six-result deterministic reconstruction and observability privacy contract.
- [Phase 17 parity](../data/evaluation/phase17_qdrant_parity.json) and
  [qualification](../data/evaluation/phase17_demo_qualification.json) —
  public-profile/local deployment evidence.
- [Tracked result table](../data/evaluation/phase16_reported_results.csv) —
  text-free aggregate source of truth.

No end-to-end “legal correctness accuracy” metric exists in this repository.
Evaluation assets include AI-reviewed/adjudicated material and engineering
diagnostics; they are not human legal-expert gold.

## Known limitations and failure modes

- Results are bounded to their named DEV or frozen evaluation populations.
- Arabic embedding and dialect findings are partial or inconclusive; no model
  promotion follows from the Phase 15 experiments.
- Full hybrid-plus-reranker latency is high on the measured local CPU profile.
- Private corpus permissions, metadata completeness, and redistribution rights
  constrain public reproduction.
- The public demo is synthetic and retrieval-first, not production-equivalent.
- Extraction coverage and semantic quality are not established as legal gold.
- A qualified human must review any real legal use.

## Safety boundaries

See the canonical [safety and limitations document](safety-and-limitations.md).
It covers jurisdiction, private data, MLflow, public demo, invalid output,
licensing, latency, and no-legal-advice boundaries in one place.

## Reproducibility and provenance

Use Python `>=3.11,<3.13`, `uv`, and the locked dependencies. Public aggregate
results can be reconstructed with `make phase16-verify`; full raw-data
experiments require authorized private/local assets and are not rerun by the
portfolio closeout. The complete MLflow database is intentionally local and
ignored; a safe text-only evidence export is tracked in
[`data/evaluation/portfolio_mlflow_evidence.json`](../data/evaluation/portfolio_mlflow_evidence.json).

## Version

The repository package version is `0.1.0`. The serving identity is content
addressed by configuration version
`3fcd52f794ca6402c5d1d6f8be3a2aa9487ea79ac8ccd967392be3304638c83c`.
Authoritative provenance files and historical reports are retained rather than
rewritten.
