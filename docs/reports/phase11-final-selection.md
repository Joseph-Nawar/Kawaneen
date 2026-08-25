# Phase 11 final selection

Status: `PHASE11_CLOSED`

## Objective and scope

Phase 11 established a narrowly regulatory extraction benchmark over the governed `saudi-moj-derived` universe. The benchmark uses structurally atomic article/clause/paragraph-sized units, document-disjoint DEV/HOLDOUT splits, deterministic candidate extraction, and a hybrid semantic layer. The protected HOLDOUT was evaluated once after its reference release was frozen.

The references are independently AI-reviewed and adjudicated. They are not human gold: `human_verified=0` throughout.

## Annotation and candidate methodology

The final benchmark contains 80 DEV and 40 protected HOLDOUT records. DEV and HOLDOUT are document-disjoint. Selection is reproducible from the selection, eligibility, corpus, and candidate-policy fingerprints.

The deterministic layer (`phase11-candidates-v3`) owns candidate detection and conservative normalization for temporal, monetary, percentage, article, and regulation references. Semantic role classification remains in the hybrid layer. Exact canonical source spans, candidate-family validation, server-side unique-occurrence resolution, field-local rejection, provenance, and Pydantic result validation are enforced server-side.

The DEV annotation release contains 77 AI-adjudicated records and three dual-AI agreements. The HOLDOUT release contains 39 AI-adjudicated records and one dual-AI agreement. Neither release is human gold.

## Hybrid experiment history

### B0

The initial B0 integration exposed a contract mismatch: otherwise useful provider responses were discarded when one field contained an invalid candidate reference or unsupported span. B0 is retained as historical evidence and is marked invalid for semantic evaluation.

### B1

B1 introduced typed candidate allowlists, field-local invalid-reference and span rejection, server-authoritative unique-occurrence resolution, and corrected provider-call accounting. Its clean DEV run completed 79/80 records. It was a valid safety diagnostic but showed severe semantic under-extraction.

### B2

B2 was the final permitted prompt-only DEV experiment. It added a small set of synthetic exact-span examples, explicit regulated-entity guidance, complete-action guidance, optional candidate classification, and separate-rule guidance. Model, schema, candidates, validator, token budget, and runtime were unchanged.

Clean B2 DEV completed 78/80 records. End-to-end results were: micro P/R/F1 `0.333/0.064/0.108`, macro F1 `0.191`, clause exact `15/80 = 18.75%`, regulated-entity F1 `0.106`, actor F1 `0.072`, action F1 `0.062`, and full-rule exact F1 `0`.

## Protected HOLDOUT procedure

The HOLDOUT reference was frozen before inference. The run used the frozen B2 configuration, exactly the original 40 protected IDs, deterministic source order, per-record persistence, and zero automatic retries. Reference labels were excluded from inference inputs and used only for this later offline evaluation. No HOLDOUT tuning, resampling, annotation change, or additional inference occurred.

The run completed 39/40 records. The one permanent failure was `f52c0c03-a024-513c-ac04-f79b16e9a234`: attempt 1 timed out; the authorized attempt 2 returned a response that failed JSON parsing with truncation evidence. Provider schema validation, exact-span validation, and candidate-reference validation were not reached. No third attempt was made.

## Final HOLDOUT result

The primary end-to-end view treats the failed record as producing no predictions and counts its reference content as false negatives. The conditional view evaluates only the 39 valid completed results.

End-to-end HOLDOUT (40 records):

- rules: gold `73`, predicted `21`;
- regulated entities: TP/FP/FN `5/10/34`, F1 `0.185`;
- actor: TP/FP/FN `5/4/39`, F1 `0.189`;
- action: TP/FP/FN `4/17/69`, F1 `0.085`;
- modality accuracy: `0.667` over 18 comparable rules;
- full-rule exact: TP/FP/FN `1/20/72`, F1 `0.021`;
- micro P/R/F1: `0.338/0.108/0.164`;
- macro F1: `0.118`;
- clause exact: `6/40 = 15.0%`.

Conditional HOLDOUT (39 completed records):

- rules: gold `64`, predicted `21`;
- regulated entities F1 `0.200`;
- actor F1 `0.222`;
- action F1 `0.094`;
- modality accuracy `0.667` over 18 comparable rules;
- full-rule exact F1 `0.024`;
- micro P/R/F1: `0.338/0.124/0.181`;
- macro F1 `0.126`;
- clause exact: `6/39 = 15.4%`.

Other end-to-end fields (TP/FP/FN; P/R/F1): conditions `6/9/22; 0.400/0.214/0.279`; exceptions `0/2/11; 0/0/0`; penalties `0/0/1; 0/0/0`; deadlines `2/0/5; 1.000/0.286/0.444`. Effective-date, monetary-threshold, and percentage-threshold supports are zero and therefore not estimable.

HOLDOUT modality counts were obligation support `48`, predicted `11`, matched `9`, F1 `0.305`; permission support `19`, predicted `4`, matched `2`, F1 `0.174`; prohibition support `6`, predicted `4`, matched `1`, F1 `0.167`. The gold-row/prediction-column comparable-rule matrix was:

| gold \\ predicted | obligation | permission | prohibition |
|---|---:|---:|---:|
| obligation | 9 | 2 | 1 |
| permission | 2 | 2 | 1 |
| prohibition | 0 | 0 | 1 |

## Safety and structural integrity

For completed outputs, final schema validity was `39/39 = 100%` and provenance completeness was `39/39 = 100%`. End-to-end pipeline completion was `39/40 = 97.5%`; raw provider-schema validity was also `39/40 = 97.5%`.

Unsupported spans were proposed 19 times and dropped 19 times; unsupported-span acceptance was `0/19 = 0%`. Invalid candidate references were proposed 4 times and dropped 4 times; invalid-candidate acceptance was `0/4 = 0%`. All hard safety gates passed.

Field-local diagnostics: 23 valid empty final results; 17 fully empty provider bodies; 17 raw entity proposals with 15 accepted; 32 raw rule proposals with 21 accepted; 11 rules dropped for invalid actions; 12 entities dropped; 15 conditions dropped; 1 exception dropped; 48 unique-occurrence corrections; 2 ambiguous/repeated occurrences rejected.

Observed error taxonomy counts were: `MISSED_EXTRACTION=46`, `SPURIOUS_EXTRACTION=3`, `WRONG_MODALITY=6`, `WRONG_ACTOR_ACTION_ASSOCIATION=3`, `WRONG_CANDIDATE_CLASSIFICATION=6`, `UNSUPPORTED_MODEL_SPAN=19`, `PIPELINE_FAILURE=1`, and zero observed duplicate, annotation-ambiguity, or separately classified span-boundary errors under the existing evaluator.

## DEV versus HOLDOUT

All values are end-to-end and use the same evaluator; HOLDOUT minus DEV is the delta.

| metric | DEV | HOLDOUT | delta |
|---|---:|---:|---:|
| completion | 78/80 (97.5%) | 39/40 (97.5%) | 0 |
| predicted rules | 31 | 21 | -10 |
| valid empty outputs | 56 | 23 | -33 |
| regulated-entity F1 | 0.106 | 0.185 | +0.079 |
| actor F1 | 0.072 | 0.189 | +0.116 |
| action F1 | 0.062 | 0.085 | +0.023 |
| full-rule exact F1 | 0.000 | 0.021 | +0.021 |
| modality accuracy | 0.739 | 0.667 | -0.072 |
| micro precision | 0.333 | 0.338 | +0.005 |
| micro recall | 0.064 | 0.108 | +0.044 |
| micro F1 | 0.108 | 0.164 | +0.056 |
| macro F1 | 0.191 | 0.118 | -0.072 |
| clause exact rate | 18.75% | 15.0% | -3.75 pp |
| unsupported proposals | 46 | 19 | -27 |
| unsupported acceptance | 0 | 0 | 0 |
| invalid candidate proposals | 5 | 4 | -1 |
| invalid candidate acceptance | 0 | 0 | 0 |

The extractor generalizes in a limited sense: HOLDOUT retains measurable semantic extraction and improves end-to-end micro, entity, actor, action, and full-rule metrics, but clause exactness and macro F1 are lower. Semantic recall remains the dominant limitation. Full-rule exact F1 becomes non-zero but remains near zero. Regulated entities are more useful than in B1, whose entity F1 was zero. Deadline classification is comparatively reliable on its small support; effective-date classification is not estimable. The main local 4B failure modes are abstention/under-extraction, incomplete actions, rejected unsupported spans, invalid references, rule loss, wrong associations/modality, and one truncated output.

## Capability status

| capability | status | basis |
|---|---|---|
| temporal/date/duration candidates | operationally supported / precision-audited | deterministic candidate generation; candidate recall and F1 not independently established |
| monetary candidates | experimental | precision audit only; recall/normalization accuracy not estimable |
| percentage candidates | experimental | precision audit only; recall not estimable |
| article references | operationally supported / precision-audited | deterministic candidate generation; candidate recall and F1 not independently established |
| regulation references | experimental | bounded precision audit; recall not estimable |
| regulated entities | experimental | measurable HOLDOUT F1, low recall |
| normative rules | experimental | measurable extraction, very low recall and near-zero full-rule exactness |
| conditions/exceptions | experimental | conditions measurable; exceptions weak |
| penalty spans | experimental | support of one, no successful match |
| deadline semantic classification | experimental | F1 `0.444` on support seven |
| effective-date semantic classification | not_evaluable | reference support zero |
| monetary/percentage semantic classification | not_evaluable | reference support zero |

## Final disposition

`PHASE11_HYBRID_EXPERIMENTAL_LIMITED`

The hybrid layer has measurable but weak semantic utility and preserved all hard source-grounding and schema-safety gates. It is not production-quality and its low recall and low full-rule exactness must remain explicit. No further tuning is authorized by this closure, and no HOLDOUT tuning occurred.

Phase 11 is closed with the selected configuration recorded as `PHASE11_CLOSED`. The final selected configuration remains the frozen B2 configuration; closure records evidence and disposition rather than changing it.

## Reproducibility hashes

The closed tracked configuration manifest records the complete final hash set. The principal inputs are:

- HOLDOUT annotation release fingerprint: `5fe335d2910bb1ea2662bdc912ec8bab4d73b15870b1a777b8a947a3315278c9`;
- HOLDOUT annotation release SHA-256: `2a5ade4acbc897bdca9456083970ada5809e8d0baf45d3ebc559472905b220b9`;
- B2 prompt hash: `72d8e6b613b56ddb61cf536df55760b2220f6e570c2f09b8536e7263d0c1f9bf`;
- schema hash: `7e1e0287c0a384d09fddc419964e3422b95a1540d2a7fc92aca58fa692d03ee0`;
- candidates-v3 hash: `dcc40496967242ee1cba99576bebce9eca3520d639f4ae25a6bb2fe0797cd675`;
- B2 DEV result SHA-256: `58d4a6b16fd8578d3b776646ee314720eed32d6c0a6350b7d51d983acef2d672`;
- B2 DEV evaluation SHA-256: `b3bda87e1791285007c780392854375bfe1431cbbf098dce0ae2468a55ae5a32`;
- B2 HOLDOUT result SHA-256: `0fba9650e08256339499438d993401d49366d0caf3b8daa071a6a254fad30f16`.

No Qwen/Ollama calls were made during closure. No HOLDOUT inference, annotation, or tuning occurred during closure.
