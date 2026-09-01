# Phase 15 protocol amendment

This text-only governance record preserves the chronology of the Phase 15 review
protocol. The original approved design specified at least 100 human engineering
adjudications after the DEV experiments. No human decisions were collected.

Before any labels could be observed, the protocol was reduced to a frozen,
pre-selected 30-case audit subset. The user subsequently removed the manual
review requirement because the time needed for 30 cases was not available. The
final audit is therefore `AUTOMATED_ADJUDICATION_DIAGNOSTIC`: a deterministic
evidence-rule pass followed by a second consistency-check pass over the frozen
case evidence. It is not human review, expert review, legal review, or human
gold.

The 120-case diagnostic population and 30-case audit identities remain frozen:

- 120-case population hash: `8bc039f51344f3af47b817a5e1bbf51d4d087f49768ea5ca2b2ce3a29cd53777`
- 30-case audit hash: `4fc44ab5f5284ed720421dd13ef6f866e69292b5709b753c5464afacc6bd8af9`

No Phase 3, Phase 8, or Phase 11 HOLDOUT was accessed or rerun. The tracked
Phase 3–14 evidence remains frozen and unchanged. This amendment reduces the
evidentiary strength of the error analysis: the automated audit is an
engineering diagnostic and the enriched 30-case subset must not be interpreted
as population prevalence or a legal-correctness estimate.

The original design and implementation-plan files remain unchanged as the
historical record. Final Phase 15 artifacts use
`AUTOMATED_ADJUDICATION_DIAGNOSTIC` and do not use the obsolete human-review
provenance label for the Phase 15 error audit.
