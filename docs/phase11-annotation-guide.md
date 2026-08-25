# Phase 11A DEV annotation guide

This guide is for the human review of the private DEV annotation pack. It does not change the Phase 11A extraction architecture or enable semantic inference.

## Pack audit

The private annotation records are individual JSON files at:

```text
artifacts/private/phase11_extraction/annotations/*.json
```

The directory contains 120 private records: 80 with `split: "dev"` and 40 with `split: "holdout"`. The DEV records are selected by the text-free manifest at `data/manifests/extraction/phase11_annotation_selection.json`; the manifest is not an annotation-editing file. Raw canonical text and annotations are private and must not be copied into tracked files.

Every DEV record contains:

- `canonical_text`, `document_id`, source provenance, and a source fingerprint.
- A populated `candidate_registry`. Each candidate has a visible `candidate_id` (`T###`, `M###`, `P###`, `A###`, or `R###`), candidate type, exact raw text, and an exact span with canonical codepoint `start_char` and `end_char` offsets.
- The current `human_annotations`, `annotation_status`, `annotation_provenance`, and `human_verified` values. The current pack has 79 unreviewed records and one existing reviewed/human-verified record; the independent-review batch forces `human_verified: false` for every exported record.
- Selection strata and the smoke flag.

In the current DEV selection, deterministic candidates are present in 32 records: 103 regulation candidates, 29 article candidates, and 1 temporal candidate. There are no DEV monetary or percentage candidates in the current pack. These counts describe candidates, not labels. A candidate is never a gold semantic classification merely because it was detected.

## Human annotation contract

The human update file has exactly this outer shape:

```json
{
  "human_annotations": {
    "schema_version": "phase11-proposal-v1",
    "regulated_entities": [],
    "rules": [],
    "exceptions": [],
    "penalties": [],
    "deadline_refs": [],
    "effective_date_refs": [],
    "monetary_threshold_refs": [],
    "percentage_threshold_refs": []
  },
  "annotation_status": "in_review",
  "human_verified": false
}
```

The update schema is strict: unknown keys are rejected. A semantic source span is an object of the form `{ "text": "...", "occurrence": 0 }`. `text` must be an exact substring of the record's canonical text. `occurrence` is optional and is used only when the same exact text occurs more than once. The validator resolves canonical Python codepoint offsets from this text; the human does not hand-enter offsets. Candidate references are strings from that record's registry, not newly invented IDs.

| Field | Human entry | Contract |
|---|---|---|
| `regulated_entity` | One or more exact source spans in `regulated_entities` | Name only what the clause literally regulates. Do not infer an entity from external law. |
| `obligation` | A `rules` item with `modality: "obligation"` | Add one rule per distinct normative rule; its `action` is an exact span. |
| `prohibition` | A `rules` item with `modality: "prohibition"` | Use only when the clause literally prohibits the action. |
| `permission` | A `rules` item with `modality: "permission"` | Use only when the clause literally permits the action. |
| `actor` | Exact span in the rule's `actor`, or `null` | Only a literal actor span is allowed. An implicit actor remains absent. |
| `action` | Required exact span in the rule's `action` | Do not paraphrase, normalize, or expand the action. |
| `conditions` | Zero or more exact spans in `rule.conditions` | Use for requirements that determine when the rule applies. |
| `exceptions` | Exact spans in `rule.exceptions` or top-level `exceptions` | Use for carve-outs or exclusions; do not use conditions as exceptions. |
| `penalty` | Exact spans in top-level `penalties` | Record the penalty wording literally. An amount in a penalty is not automatically a monetary threshold. |
| `deadline` | Existing `T###` in `deadline_refs` (top-level or on the rule) | Classify a detected temporal candidate only when the clause states a deadline or duration. Do not add a date candidate. |
| `effective date` | Existing `T###` in `effective_date_refs` (top-level or on the rule) | Classify only when the clause states when a rule takes effect. An ordinary date remains unclassified. |
| `monetary threshold` | Existing `M###` in `monetary_threshold_refs` (top-level or on the rule) | Classify only when the clause uses the amount as a threshold or condition. |
| `percentage threshold` | Existing `P###` in `percentage_threshold_refs` (top-level or on the rule) | Classify only when the percentage is a threshold or condition. |
| article/regulation reference | Leave deterministic `A###`/`R###` unchanged | The current human schema does not permit correction or replacement of deterministic article/regulation candidates. Do not invent an ID. |
| `issuing_authority` | No human entry | It is governed metadata only and remains null when unavailable. |
| negative/no-target clause | Empty `SemanticProposal` arrays and reference lists | This is valid. Review it as a negative record; do not create a rule for descriptive or factual text. |

`ProposedRule` has the exact fields `modality`, `actor`, `action`, `conditions`, `exceptions`, `deadline_refs`, `effective_date_refs`, `monetary_threshold_refs`, and `percentage_threshold_refs`. The outer proposal has the fields shown above. `obligation`, `prohibition`, and `permission` are not separate arrays; they are the three allowed modality enum values.

Use `annotation_status: "in_review"` while a record is not fully checked. Set `annotation_status: "reviewed"` only after the entire record has been reviewed. Set `human_verified: true` only at that same point. A reviewed record may intentionally retain `human_verified: false` until the annotator is ready to declare it verified.

## Annotation principles

- Use exact canonical text spans only. Preserve the original Arabic codepoints; never paraphrase or destructively normalize.
- Annotate what the clause itself states, not what external law might imply.
- Do not infer missing actors, penalties, dates, durations, thresholds, or authorities.
- An implicit actor with no literal source span remains absent; do not manufacture an actor marker.
- Phase 11A is regulatory only. Do not label ordinary descriptive, factual, or case-law language.
- A clause can contain zero, one, or multiple normative rules.
- Conditions and exceptions are different: conditions define applicability; exceptions carve out a case otherwise covered by the rule.
- A detected money or date candidate is not automatically a monetary threshold, effective date, or deadline.
- `issuing_authority` remains metadata-only and must not be manually invented.
- Negative/no-target clauses are valid annotations with no semantic rules.
- Unsupported or ambiguous spans must be omitted rather than guessed.

## Synthetic examples

The following examples are synthetic and are not corpus text. Candidate IDs shown in an example assume that the corresponding candidate exists in that example's private registry.

1. Obligation, actor, and deadline:

```json
{
  "text": "يلتزم المرخص له بتقديم الطلب خلال ٣٠ يوماً.",
  "human_annotations": {
    "schema_version": "phase11-proposal-v1",
    "regulated_entities": [{"text": "المرخص له"}],
    "rules": [{"modality": "obligation", "actor": {"text": "المرخص له"}, "action": {"text": "تقديم الطلب"}, "conditions": [], "exceptions": [], "deadline_refs": ["T001"], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": []}],
    "exceptions": [], "penalties": [], "deadline_refs": ["T001"], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": []
  }
}
```

2. Prohibition:

```json
{
  "text": "يحظر على المنشأة إفشاء البيانات.",
  "human_annotations": {
    "schema_version": "phase11-proposal-v1",
    "regulated_entities": [],
    "rules": [{"modality": "prohibition", "actor": {"text": "المنشأة"}, "action": {"text": "إفشاء البيانات"}, "conditions": [], "exceptions": [], "deadline_refs": [], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": []}],
    "exceptions": [], "penalties": [], "deadline_refs": [], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": []
  }
}
```

3. Permission:

```json
{
  "text": "يجوز للجهة تمديد المهلة.",
  "human_annotations": {
    "schema_version": "phase11-proposal-v1",
    "regulated_entities": [],
    "rules": [{"modality": "permission", "actor": {"text": "للجهة"}, "action": {"text": "تمديد المهلة"}, "conditions": [], "exceptions": [], "deadline_refs": [], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": []}],
    "exceptions": [], "penalties": [], "deadline_refs": [], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": []
  }
}
```

4. Exception:

```json
{
  "text": "يلتزم المرخص بالحفظ، ما لم يصدر استثناء مكتوب.",
  "human_annotations": {
    "schema_version": "phase11-proposal-v1",
    "regulated_entities": [],
    "rules": [{"modality": "obligation", "actor": {"text": "المرخص"}, "action": {"text": "الحفظ"}, "conditions": [], "exceptions": [{"text": "ما لم يصدر استثناء مكتوب"}], "deadline_refs": [], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": []}],
    "exceptions": [], "penalties": [], "deadline_refs": [], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": []
  }
}
```

5. Penalty with an amount, without automatic threshold classification:

```json
{
  "text": "يعاقب المخالف بغرامة قدرها ٥٠٠ ريال.",
  "human_annotations": {
    "schema_version": "phase11-proposal-v1",
    "regulated_entities": [], "rules": [],
    "exceptions": [], "penalties": [{"text": "غرامة قدرها ٥٠٠ ريال"}],
    "deadline_refs": [], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": []
  }
}
```

6. Percentage threshold:

```json
{
  "text": "يشترط ألا تقل النسبة عن ١٥٪.",
  "human_annotations": {
    "schema_version": "phase11-proposal-v1",
    "regulated_entities": [],
    "rules": [{"modality": "obligation", "actor": null, "action": {"text": "ألا تقل النسبة عن ١٥٪"}, "conditions": [], "exceptions": [], "deadline_refs": [], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": ["P001"]}],
    "exceptions": [], "penalties": [], "deadline_refs": [], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": ["P001"]
  }
}
```

7. Effective date versus ordinary date:

```json
{
  "text": "يسري القرار اعتباراً من ١/١/١٤٤٥هـ.",
  "human_annotations": {
    "schema_version": "phase11-proposal-v1",
    "regulated_entities": [], "rules": [], "exceptions": [], "penalties": [],
    "deadline_refs": [], "effective_date_refs": ["T001"], "monetary_threshold_refs": [], "percentage_threshold_refs": []
  }
}
```

For comparison, `ورد في التقرير تاريخ ١/١/١٤٤٥هـ.` is an ordinary descriptive date: use an empty proposal and do not put `T001` in `effective_date_refs`.

8. Negative/no-target clause:

```json
{
  "text": "يتناول النص تعريف المصطلح فقط.",
  "human_annotations": {
    "schema_version": "phase11-proposal-v1",
    "regulated_entities": [], "rules": [], "exceptions": [], "penalties": [],
    "deadline_refs": [], "effective_date_refs": [], "monetary_threshold_refs": [], "percentage_threshold_refs": []
  }
}
```

## Local workflow

The smallest practical helper is intentionally local and file-based:

```text
uv run kawaneen extraction annotate-dev --next
uv run kawaneen extraction annotate-dev --save --record-id <canonical_unit_id> --annotation-file <private-update.json>
uv run kawaneen extraction annotation-progress --split dev
uv run kawaneen extraction validate-annotations --split dev
```

`--next` displays one unreviewed DEV record, including its private canonical text and candidate registry, for the annotator. The update file should be kept private. `--save` accepts only one DEV `canonical_unit_id`, validates the update with the existing annotation validator, and atomically replaces that record. It never calls a model, never accesses HOLDOUT, and never writes tracked source text. There is no HOLDOUT option on these helper commands.

For terminal-guided review without creating an update file, use exactly one interactive record per invocation:

```text
uv run kawaneen extraction annotate-dev --next --interactive
```

The interactive mode displays the record position, private canonical text, grouped deterministic candidates, and any existing `in_review` proposal. It asks for exact spans, occurrence indices only when a span is duplicated, normative rules, and candidate references. It validates before saving, supports the fast empty-proposal path, asks for explicit reviewed or unfinished confirmation, saves atomically, reports updated progress, and prints the command for the next record without starting it automatically.

Progress fields are `total`, `reviewed`, `human_verified`, `remaining`, and `invalid`. The validator remains the authoritative check and must be run with `uv run kawaneen extraction validate-annotations --split dev` after edits.

## Independent-AI batch review

The complete private DEV batch is exported to:

```text
artifacts/private/phase11_extraction/review/phase11_dev_annotation_batch_v1.json
```

It contains exactly the 80 DEV records and no HOLDOUT records. Its top-level contract identifies `phase11-proposal-v1`, the allowed modalities, exact `{text, occurrence}` spans, candidate-reference rules, target fields, and `phase11-annotation-contract-v1`. Export is available through:

```text
uv run kawaneen extraction export-dev-annotation-batch
```

An explicitly supplied independent-AI result can eventually be imported with:

```text
uv run kawaneen extraction import-reviewed-dev --file <private-reviewed-json>
```

Use `--partial` only for an intentional partial import. The importer requires the exact DEV selection fingerprint, rejects unknown/HOLDOUT, duplicate, or missing IDs in full mode, validates exact spans and candidate references with the existing validator, preserves immutable source/candidate data, writes atomically, and records `annotation_provenance: "independent_ai_review"`. It always keeps `human_verified: false`.

`annotation_provenance` is additive and backward-compatible for existing records that omit it. Supported future provenance states are `unreviewed`, `independent_ai_review`, `dual_ai_agreed`, `dual_ai_disagreement`, and `human_adjudicated`; only a genuinely human-adjudicated record can be human-verified.
