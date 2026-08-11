# Statutory reconciliation

The Saudi Ministry of Justice Legal Portal, Saudi Bureau of Experts legislation portal,
and Saudi National Platform rules catalogue are manual verification authorities. Phase
3 performs no bulk scraping and does not infer official title, status, dates, amendment
history, article counts, or reuse rights from the derived seed.

The 3,185-row MOJ-derived seed contains 71 law names. The initial 1,993 duplicate
law/article-key statistic was produced by the superseded partial-label parser. After
the corrected full structural-label parser, the derived layer contains 949 structural
groups, 450 duplicate groups, and 2,236 duplicate-key excess rows. Every row is
retained as a `SourceFragment`. Unique groups remain unique; only explicit
part-labelled series are mergeable. Conflicting and unresolved groups remain
separately represented with reconstruction status and fragment IDs.

`data/manifests/reconciliation/core-commercial-civil-v1.csv` is a typed manual-review
record. Controlled `not_verified` and `not_reviewed` values distinguish unavailable
evidence from a measured zero; neither establishes eligibility. The current candidate
list is litigation-heavy. Later curation should
prioritize domain coverage and authoritative records for companies/business formation,
digital commerce, and other core commercial instruments rather than maximizing the
raw number of titles.

The sanitized gap report identifies 20 targeted commercial/civil instruments. It is a
planning artifact, not an acquisition approval. An official machine-readable export
with stable article identifiers and explicit access, processing, quotation,
redistribution, and public-display terms remains the preferred replacement.

The corrected duplicate diagnostic records 949 groups, 450 duplicate groups, and
2,686 rows inside those groups. It reports 3,174 high-confidence labels, 11
unresolved labels, 25 ambiguous continuation candidates, and 23 explicit
fragment-series groups. No ambiguous group is merged. The superseded 1,192/446/2,439
values remain only as historical baseline values in the Phase 3 report and canonical
diagnostic metadata. A deterministic 25-group review sample is committed as metadata
only; no source text is included.

`groups` is a derived structural metric, not an official article count: it counts
unique `(law_name, full structural article-label key)` groups after parsing. Explicit
part-labelled rows share the parent article group while retaining their part index;
unresolved labels receive row-specific groups and cannot merge by partial similarity.
Thus 95 source fragments for `نظام المحاكم التجارية` yielding 27 groups means 27
derived label groups under this conservative structural rule, not 27 authoritative
active articles. The private 12-law audit
`artifacts/private/handoff/phase3-independent-review/phase3_statutory_structural_audit.csv`
contains one sanitized metadata row per core-law fragment and records the passed
ordinal/part non-collision assertions.

## Completed external reconciliation decision

The supplied external adjudication was validated against the frozen 12-law candidate
set. It is `independent_ai_source_review`, not human or legal-expert verification.
All twelve laws have the controlled final outcome
`present_partial_reviewed_not_eligible`; zero laws are eligible for the trusted v1
statutory corpus. The sanitized reconciliation CSV retains the external artifact
digest and explicit `human_verified=false` provenance.

The historical review handoff had 48/60 target-presence coverage because an older
exporter used partial ordinal grouping. The regenerated private handoff selects by
law and full parsed ordinal, has 60/60 target presence, and permits only part/status
variants of the selected ordinal in a target. Regeneration does not change the
negative statutory eligibility decision.
