# Phase 10 Stage A.1 handoff — real-generator readiness

Status: ready for the user-controlled manual Qwen DEV sequence after the local Ollama model is pulled and locked. No model inference, NLI, retrieval, holdout, full-suite test, coverage run, `make check`, commit, push, or Phase 11 work was performed.

## 1. Jurisdiction audit

The authoritative governed evidence is:

- `data/manifests/source_registry.csv` — SHA-256 `8d5e9a6be1ae3083c8d23574d330b53b9baecfb7050bbfb1a8945fb10aac4ae2`; its active-release source records identify ALARB and ArabiCCR as Saudi Arabia sources.
- `data/manifests/evaluation/phase6_ai_reviewed_v1_manifest.json` — SHA-256 `1b50e262fce2a0e09f75b9ad4d7bf14bf24bfaea0247f18f87e875f0d4bf6ddd`; its frozen release contains only `alarb` and `arabiccr` as the primary evaluation sources.

These repository metadata/configuration artifacts explicitly establish the Saudi v1 deployment scope. The derived contract is `data/manifests/generation/phase10_jurisdiction_scope.json`:

- active jurisdiction: `SA`
- allowed jurisdictions: `{SA}`
- mode: `single`
- status: active

Runtime behavior is fail closed. An explicit non-SA jurisdiction returns `JURISDICTION_MISMATCH`; conflicting jurisdiction markers or conflicting context jurisdiction returns `JURISDICTION_AMBIGUOUS`; an unresolved required scope abstains as `JURISDICTION_AMBIGUOUS`. The server injects the jurisdiction into the prompt and the strict model-output contract rejects jurisdiction metadata, so generator output cannot override policy. English and Arabic Saudi/Egyptian cases are covered by focused tests.

## 2. Retrieval-sufficiency calibration

The persisted DEV-only one-dimensional top-reranker-score study selected no threshold. There were zero cutoffs satisfying the frozen precision `>=0.95` and minimum allowed-query count `>=10` rule. Selected threshold, precision, and recall are therefore `null`; score gating remains disabled. With gating disabled, all 160 DEV queries are allowed through. No cutoff was invented or retuned.

## 3. Deterministic extractive DEV baseline

The persisted baseline is unchanged. Deterministic extractive mode is now explicitly `benchmark_only: true`, with no automatic fallback on Qwen failure, timeout, invalid JSON, invalid citations, unsupported claims, or unavailable Qwen; those paths abstain.

Populations:

1. Answerable + gold in supplied top-8: **38**
2. Answerable + gold absent from supplied top-8: **103**
3. Explicitly unanswerable: **19**

| Metric | Value |
|---|---:|
| SupportedAnswerPrecision | 12/146 = 0.0821917808 |
| SupportedAnswerCoverage | 12/141 = 0.0851063830 |
| ContextInsufficientAbstentionRecall | 3/103 = 0.0291262136 |
| UnanswerableAbstentionRecall | 7/19 = 0.3684210526 |
| FalseAnswerRate | 12/19 = 0.6315789474 |
| FalseAbstentionRate | 4/38 = 0.1052631579 |
| ValidCitationRate | 257/257 = 1.0000000000 |
| ClaimCitationCoverage | 257/257 = 1.0000000000 |
| GoldCitationHitRate | 12/257 = 0.0466926070 |
| CompleteGoldEvidenceUse | 12/38 = 0.3157894737 |

Counts: 0 invalid generations, 0 unsupported claims, 12 policy-gated abstentions (`CURRENTNESS_UNVERIFIED` 6; `NO_CONTEXT` 6), 257 claims, 257 citations, and 257 valid citations.

## 4. Token-budget status

- Qwen tokenizer model ID: `Qwen/Qwen3-4B-Instruct-2507`
- immutable tokenizer revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- configured total input cap: **3584**
- output reservation: **384**
- safety margin: **128**
- fixed prompt/schema overhead: not computable before the real tokenizer is locally available
- resulting evidence budget: not computable before the real tokenizer is locally available
- Phase-9 assembly exercised with the real tokenizer: **no**
- `BudgetedGoldEvidenceRetention`: `null`
- `BudgetedCompleteGoldEvidenceRetention`: `null`

The tokenizer and Qwen model use the same repository revision. The persisted codepoint budget audit remains clearly marked legacy and is not reported as a Qwen result.

## 5. Model registry and locks

- Qwen HF model: `Qwen/Qwen3-4B-Instruct-2507`
- immutable model revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- tokenizer ID/revision: same repository and same full revision above
- Fanar: `QCRI/Fanar-1-9B-Instruct`, revision unresolved; optional and non-blocking
- Ollama tag: `qwen3:4b-instruct-2507-q4_K_M`
- Ollama digest: unresolved pending the user’s local pull; it is not invented or written into the tracked lock
- generator lock records `weights_downloaded_or_loaded: false`

The local Ollama lock is private and text-free at `artifacts/private/phase10_generation/qwen-ollama-model-lock.json`. The lock flow queries `/api/tags`, requires the exact tag and a full local digest, and generation refuses a tag/digest mismatch.

## 6. Manual commands — do not execute as part of this handoff

Run these in order:

```sh
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

```sh
uv run kawaneen generation lock-ollama \
  --model qwen3:4b-instruct-2507-q4_K_M \
  --endpoint http://localhost:11434 \
  --lock-path artifacts/private/phase10_generation/qwen-ollama-model-lock.json
```

```sh
uv run kawaneen generation status --generator qwen-ollama
```

```sh
uv run kawaneen generation run-dev --generator qwen-ollama --resume
```

The runner uses persisted Phase-8 rankings and private Phase-9 ContextPacks, performs production-tokenizer budgeting lazily, never loads qrels into its runtime query contract, writes atomic per-query private results/checkpoints, validates completed checkpoints on resume, and fingerprints the required query/model/tokenizer/prompt/policy/context/decode inputs.

## 7. Stage-A artifacts

The tracked artifact paths and SHA-256 values are returned with this handoff. The jurisdiction contract is an additional tracked Stage-A artifact. The private Ollama lock is intentionally absent until the manual pull/lock command succeeds.

## 8. Verification

- focused generation/grounding tests: **97 passed**
- Ruff on generation, CLI, generation tests, and grounding tests: **passed**
- strict generation/grounding Pyright: **0 errors, 0 warnings, 0 informations**
- tracked/private text audit: **passed**
- deterministic/hash audit: **passed**

Stage A.1 is ready for the manual Qwen DEV sequence. This handoff stops before any model inference.
