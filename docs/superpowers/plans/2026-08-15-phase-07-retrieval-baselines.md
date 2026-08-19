# Phase 7 Retrieval Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish reproducible keyword, BM25, and exact dense retrieval baselines over the immutable Phase 6 AI-reviewed evaluation release, with frozen dev selection and one-shot holdout reporting.

**Architecture:** Add a focused `kawaneen.retrieval` package that consumes Phase 5 `LegalChunk` records and Phase 6 `DatasetItem` records through read-only loaders. Keep representations, scoring, exact vector indexes, metrics, slices, latency, manifests, and orchestration in separate typed modules. CLI orchestration writes text-bearing rankings only below the ignored Phase 7 private artifact root and writes sanitized hashes/metrics/manifests to tracked paths.

**Tech Stack:** Python 3.12 (`>=3.11,<3.13`), existing Pydantic/dataclass/config/CLI conventions, `bm25s`, `sentence-transformers`, optional `faiss-cpu`, NumPy reference search, pytest/coverage, Ruff, Pyright, pre-commit, and `uv`.

## Global Constraints

- Preserve all Phase 0–6 source, corpus, normalization, chunking, evaluation, qrel, query, label, and frozen-release artifacts.
- Use only the frozen `phase6-retrieval-eval-ai-reviewed-v1` release and Phase-5 `legal-structure-v1` chunks.
- Never use retrieval results to mutate queries, qrels, chunks, labels, or selection manifests.
- Keep text-bearing outputs under `artifacts/private/phase7_retrieval/`; tracked outputs are text-free.
- Never download Hugging Face models from normal tests; real-model smoke is a separate explicit local command.
- Use deterministic tie-breaking by `chunk_id`, exact cosine/IP search, fixed BM25 parameters, seed `20260815`, and 2,000 paired-bootstrap replicates.
- Do not implement hybrid retrieval, fusion, reranking, generation, RAG, or Phase 8 work.

---

### Task 1: Add Phase 7 configuration and dependency boundaries

**Files:**
- Create: `configs/retrieval/phase7_baselines.toml`
- Modify: `pyproject.toml`
- Test: `tests/test_retrieval_config.py`

**Interfaces:**
- `load_phase7_config(path: Path = ...) -> Phase7Config`
- `Phase7Config` exposes fixed metrics, policies, BM25 parameters, model contracts, bootstrap seed/replicates, and private/tracked paths.

- [ ] Write tests for fixed config values, supported Python/dependency extras, and import-time absence of filesystem/network side effects.
- [ ] Run the focused test and confirm it fails because the retrieval config module is absent.
- [ ] Implement typed config loading from TOML with explicit path access and optional/development retrieval dependencies.
- [ ] Run focused tests and Ruff on changed files.

### Task 2: Define corpus/release contracts and private loaders

**Files:**
- Create: `src/kawaneen/retrieval/models.py`
- Create: `src/kawaneen/retrieval/corpus.py`
- Create: `src/kawaneen/retrieval/manifests.py`
- Test: `tests/test_retrieval_corpus.py`
- Test: `tests/test_retrieval_manifests.py`

**Interfaces:**
- `RetrievalChunk` preserves `chunk_id`, exact `display_text`, derived `search_text`, source/document metadata, policy hashes, and token count.
- `RetrievalRelease` contains frozen `DatasetItem` records, chunks, qrels, corpus/release hashes, and split access guards.
- `load_phase7_release(root: Path, *, allow_holdout: bool = False) -> RetrievalRelease` verifies Phase 6 manifest hashes, chunk policy, ALARB/ArabiCCR scope, qrel-to-corpus membership, and duplicate chunk rejection.
- `build_corpus_manifest(chunks, release, config) -> dict[str, object]` is deterministic and text-free.

- [ ] Add fixtures for a synthetic chunk/release and tests for release hash mismatch, missing qrel chunks, duplicate chunks, exact display text preservation, and holdout permission.
- [ ] Run focused tests and observe missing-interface failures.
- [ ] Implement read-only JSONL/JSON/Phase-5 chunk loading and deterministic hash validation; reject any mutated frozen release.
- [ ] Implement text-free corpus manifest payloads with corpus hash, legal-structure policy hash, normalization policy hashes, counts, and chunk IDs hash only.
- [ ] Run focused tests.

### Task 3: Implement tokenization and raw/light representations

**Files:**
- Create: `src/kawaneen/retrieval/tokenization.py`
- Test: `tests/test_retrieval_tokenization.py`

**Interfaces:**
- `tokenize_retrieval(text: str) -> tuple[str, ...]` reuses the existing Phase-4 Unicode-word/single-punctuation tokenizer.
- `represent(text: str, policy_id: str) -> RetrievalRepresentation` derives normalized text and tokens without mutating display text.

- [ ] Test tokenizer consistency against Phase-4 `tokenize`, raw/light output, punctuation behavior, and exact display-text preservation.
- [ ] Run focused tests and verify red.
- [ ] Implement the thin adapter over Phase-4 tokenizer and normalization policies.
- [ ] Run focused tests and existing normalization tokenization tests.

### Task 4: Implement deterministic keyword and BM25 retrieval

**Files:**
- Create: `src/kawaneen/retrieval/keyword.py`
- Create: `src/kawaneen/retrieval/bm25.py`
- Test: `tests/test_retrieval_keyword.py`
- Test: `tests/test_retrieval_bm25.py`

**Interfaces:**
- `KeywordIndex.build(chunks, policy_id) -> KeywordIndex`; `.search(query, top_k) -> tuple[ScoredChunk, ...]` uses Jaccard over token sets and `chunk_id` tie-breaking.
- `BM25Index.build(chunks, policy_id, k1=1.2, b=0.75) -> BM25Index`; `.search(query, top_k) -> tuple[ScoredChunk, ...]` uses `bm25s` and deterministic mapped ordering.
- Both expose full scores for diagnostics and include query normalization/tokenization in timed search.

- [ ] Add Jaccard and BM25 tiny-corpus tests, including ties, zero-overlap, duplicate rejection, and expected ranking.
- [ ] Run focused tests and verify red.
- [ ] Implement the minimal deterministic indexes with explicit raw/light representations and fixed parameters.
- [ ] Run focused tests and compare BM25 output with a tiny hand-computed fixture.

### Task 5: Implement dense adapters and exact vector backends

**Files:**
- Create: `src/kawaneen/retrieval/dense_models.py`
- Create: `src/kawaneen/retrieval/vector_index.py`
- Create: `src/kawaneen/retrieval/cache.py`
- Test: `tests/test_retrieval_dense.py`
- Test: `tests/test_retrieval_vector_index.py`
- Test: `tests/test_retrieval_cache.py`

**Interfaces:**
- `DenseModelAdapter` defines immutable model ID/revision, formatting, max length, batch size, `encode_queries`, `encode_passages`, and token diagnostics.
- `E5SmallAdapter` formats `query: {text}` / `passage: {text}` and uses max length 512, batch 32.
- `BGEM3Adapter` uses unprefixed text, dense-only encoding, and batch 4.
- `NumpyExactIndex` and `FaissExactIndex` validate finite float32 L2-normalized vectors and return deterministic top-k rankings.
- `embedding_cache_fingerprint(...) -> str` includes every required cache input and rejects mismatches.

- [ ] Add mocked adapter tests for E5/BGE formatting, dimensions, normalized finite vectors, token truncation diagnostics, and deterministic batch-halving resolution.
- [ ] Add vector tests for NumPy/Faiss parity, NaN/Inf rejection, normalization validation, and deterministic ties.
- [ ] Add cache invalidation tests for every fingerprint field.
- [ ] Run focused tests and confirm red.
- [ ] Implement adapters with lazy imports/model loading only inside explicit execution, exact vector backends, and private cache metadata.
- [ ] Run focused tests without downloading models.

### Task 6: Implement pure IR/evidence metrics, bootstrap, slices, and latency

**Files:**
- Create: `src/kawaneen/retrieval/metrics.py`
- Create: `src/kawaneen/retrieval/evidence.py`
- Create: `src/kawaneen/retrieval/slices.py`
- Create: `src/kawaneen/retrieval/latency.py`
- Test: `tests/test_retrieval_metrics.py`
- Test: `tests/test_retrieval_slices.py`
- Test: `tests/test_retrieval_latency.py`

**Interfaces:**
- Pure functions: `recall_at_k`, `mrr_at_k`, `ndcg_at_k`, `precision_at_k`, `complete_evidence_recall_at_k`, `paired_bootstrap`, `wins_ties_losses`.
- `assign_slices(item, query_length_bins) -> dict[str, str]` covers all required dimensions and frozen tertile bins.
- `LatencySummary.from_samples(samples) -> ...` computes p50/p95 and records device/package/thread metadata.

- [ ] Add fixtures covering binary/graded relevance, multiple evidence groups, unanswerables exclusion, frozen query-length bins, variants, and bootstrap determinism.
- [ ] Run focused tests and verify red.
- [ ] Implement metrics with binary grade > 0, graded gain `2**rel - 1`, all required groups for complete evidence, and answerable-only denominators.
- [ ] Implement required slice labels/counts, parent mapping, degradation summaries, complementarity, and latency summaries.
- [ ] Run focused tests.

### Task 7: Implement experiment orchestration and holdout freeze protection

**Files:**
- Create: `src/kawaneen/retrieval/evaluation.py`
- Create: `src/kawaneen/retrieval/orchestrator.py`
- Modify: `src/kawaneen/cli.py`
- Test: `tests/test_retrieval_evaluation.py`
- Test: `tests/test_retrieval_cli.py`

**Interfaces:**
- `retrieval_plan()`, `build_retrieval_corpus()`, `retrieval_smoke()`, `evaluate_dev()`, `freeze_dev_selection()`, `evaluate_holdout(*, allow_holdout: bool)`, and `retrieval_report()` map to the required CLI commands.
- Dev evaluates keyword/BM25/E5 raw/light and BGE using selected dense policy; freeze writes immutable text-free selection and model/config hashes.
- Holdout refuses without `--allow-holdout`, loads only frozen configurations, and does not write per-query/error-analysis outputs.

- [ ] Add mocked end-to-end tests for smoke, dev selection, frozen config immutability, holdout protection, and text-free tracked output audits.
- [ ] Run focused tests and verify red.
- [ ] Implement staged private/tracked output writes, model revision resolution/lock manifest, dense diagnostics, failures/complementarity, and no holdout tuning.
- [ ] Add the six CLI subcommands with explicit holdout flag handling.
- [ ] Run focused tests and CLI help/command tests.

### Task 8: Add tracked manifests/docs and local real-model smoke

**Files:**
- Create: `data/manifests/retrieval/phase7_model_lock.json`
- Create: `data/manifests/retrieval/phase7_corpus_manifest.json`
- Create: `data/manifests/retrieval/phase7_dev_selection.json`
- Create: `data/manifests/retrieval/phase7_final_manifest.json`
- Create: `data/evaluation/phase7_dev_metrics.json`
- Create: `data/evaluation/phase7_holdout_metrics.json`
- Create: `data/evaluation/phase7_baseline_comparison.json`
- Create: `docs/phases/phase-07-retrieval-baselines.md`
- Test: `tests/test_retrieval_artifacts.py`

- [ ] Add tests rejecting query/evidence/retrieved text in tracked outputs and checking deterministic hashes/manifests.
- [ ] Run focused tests and verify red.
- [ ] Implement text-free artifact serializers, report schema, and explicit local real-model smoke entry point that is excluded from normal CI.
- [ ] Run deterministic synthetic rebuild checks and artifact audits.

### Task 9: Execute the full Phase 7 experiment and verify the repository

**Files:**
- Modify only generated Phase 7 tracked/private outputs as required by the experiment.

- [ ] Run `uv sync` with the retrieval development dependencies.
- [ ] Run all unit/integration tests, coverage, Ruff, Pyright, pre-commit, and `make check`.
- [ ] Run `kawaneen retrieval plan`, `build-corpus`, `smoke`, `evaluate-dev`, `freeze-dev-selection`, and `evaluate-holdout --allow-holdout` in order.
- [ ] Run deterministic manifest/config rebuild checks and private/tracked artifact audits.
- [ ] Run the local real-model smoke for E5 and BGE-M3; record resolved revisions, dimensions, truncation, device, build time, and artifact sizes.
- [ ] Review the final report for every required metric/slice/comparison and confirm no Phase 8 features were added.
- [ ] Verify the pre-existing stash remains unchanged and the final diff contains no prior-phase files.
