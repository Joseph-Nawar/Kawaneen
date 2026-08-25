# Phase 11A structured regulatory extraction

Phase 11A prepares a private, measurable extraction workflow for eligible Saudi statutory/regulatory canonical units. It does not claim extraction quality: the annotation pack starts unreviewed and no semantic performance is reported until records are human-reviewed and `human_verified=true`.

## Architecture and responsibility

`deterministic-v1` builds a request-local candidate registry from canonical text. It detects temporal/date/duration, monetary, percentage, article, and regulation references, and preserves exact Arabic text and codepoint offsets. It does not decide whether a date is an effective date or deadline, or whether an amount is a threshold; those classifications require semantic context.

`hybrid-qwen-v1` adds a compact semantic proposal layer. The provider may return only exact source text for regulated entities, actors/actions/conditions, exceptions, penalties, and normative modalities, plus references to deterministic candidate IDs. It cannot return authority, source metadata, URLs, offsets, paraphrases, arbitrary normalized values, or invented dates/amounts. Every proposal is validated against the canonical text before entering the final result. Invalid or ambiguous fields are dropped with diagnostics.

## Source grounding and schema

Every final result carries `schema_version`, `extractor_version`, configuration, `SA` jurisdiction, source provenance, source fingerprint, governed issuing authority, regulated entities, modality-grouped normative rules, classified dates/thresholds, references, validation metadata, candidate registry, and field-level provenance. An exact span is valid only when:

```text
canonical_text[start_char:end_char] == text
```

No fuzzy reconstruction is permitted. Issuing authority is metadata-only and remains null when governed metadata does not provide it.

## Annotation and holdout protocol

The Phase-11 v1 universe is the governed primary statutory corpus, represented locally by canonical statutory unit types. Case-law units are ineligible through governed source role/type and canonical unit typing. The selection is 120 unique units: 80 DEV, 40 protected HOLDOUT, and a 10-unit smoke subset of DEV. DEV and HOLDOUT documents are disjoint, selection is seeded by the corpus fingerprint, and exact duplicate text hashes are avoided where available.

Raw text and annotations live under `artifacts/private/phase11_extraction/annotations/`. Tracked manifests under `data/manifests/extraction/` and reports under `data/evaluation/` contain IDs, hashes, counts, strata, and states only. Weak cues are sampling strata, never labels. HOLDOUT commands require `--allow-holdout`; ordinary DEV commands cannot access it.

## Metrics

Evaluation uses strict source-span exact match with per-field TP/FP/FN/support/precision/recall/F1, micro and macro F1, clause exact match, normative actor/action/modality/full-rule metrics, and normalized structured equality where applicable. Engineering metrics include raw/final schema validity, unsupported-span proposal and acceptance, invalid candidate-reference proposal and acceptance, and provenance completeness. Error categories include missed/spurious extraction, boundary error, wrong modality, association, candidate classification, normalization, duplicates, unsupported spans, missing metadata, and ambiguity.

## Checkpoints and limitations

Extraction checkpoints explicitly track incomplete/complete lifecycle milestones and fingerprint the source unit, configuration, candidate normalizer, prompt, schema, Qwen model/digest, tokenizer revision, and semantic policy. Readiness artifacts are not completed hybrid results. Qwen/Ollama inference, model downloads, training, NLI, holdout evaluation, and human-gold reporting are outside Phase 11A.
