"""Single-record private DEV annotation terminal workflow."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

from kawaneen.extraction.annotation import AnnotationRecord, AnnotationUpdate
from kawaneen.extraction.contracts import (
    Candidate,
    Modality,
    ProposedRule,
    ProposedSpan,
    SemanticProposal,
)
from kawaneen.extraction.orchestration import (
    annotation_progress,
    next_dev_annotation_context,
    save_dev_annotation_update,
    validate_dev_annotation_update,
)
from kawaneen.extraction.span_validation import resolve_exact_span


class InteractiveAnnotationError(ValueError):
    """The interactive session could not produce a saveable annotation."""


class InputFunction(Protocol):
    def __call__(self, prompt: str) -> str: ...


OutputFunction = Callable[[str], None]
NextContext = Callable[[], tuple[AnnotationRecord, int, int]]
ValidateUpdate = Callable[[str, AnnotationUpdate], list[str]]
SaveUpdate = Callable[[str, AnnotationUpdate], dict[str, object]]
ProgressFunction = Callable[[str], dict[str, int | str]]

_PREFIXES = ("T", "M", "P", "A", "R")
_PREFIX_LABELS = {
    "T": "temporal/deadline/effective-date",
    "M": "monetary",
    "P": "percentage",
    "A": "article",
    "R": "regulation",
}


def _terminal_input(prompt: str) -> str:
    return input(prompt)


def _ask(input_fn: InputFunction, output_fn: OutputFunction, prompt: str) -> str:
    del output_fn
    return input_fn(prompt).strip()


def _yes_no(input_fn: InputFunction, output_fn: OutputFunction, prompt: str) -> bool:
    while True:
        answer = _ask(input_fn, output_fn, f"{prompt} [y/n] ")
        if answer.casefold() in {"y", "yes"}:
            return True
        if answer.casefold() in {"n", "no"}:
            return False
        output_fn("Please answer y or n.")


def _count(input_fn: InputFunction, output_fn: OutputFunction, label: str) -> int:
    while True:
        answer = _ask(input_fn, output_fn, f"Number of {label} [0] ")
        if answer == "":
            return 0
        try:
            value = int(answer)
        except ValueError:
            output_fn("Please enter a non-negative integer.")
            continue
        if value >= 0:
            return value
        output_fn("Please enter a non-negative integer.")


def _occurrences(text: str, source: str) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    while True:
        found = text.find(source, cursor)
        if found < 0:
            return offsets
        offsets.append(found)
        cursor = found + 1


def _span(
    canonical_text: str,
    input_fn: InputFunction,
    output_fn: OutputFunction,
    label: str,
) -> ProposedSpan:
    while True:
        source = _ask(input_fn, output_fn, f"Exact source span for {label}: ")
        if not source:
            output_fn("Span rejected: proposed source span is empty")
            continue
        matches = _occurrences(canonical_text, source)
        if not matches:
            output_fn("Span rejected: proposed source span not found in canonical text")
            continue
        occurrence: int | None = None
        if len(matches) > 1:
            output_fn(f"This span occurs {len(matches)} times; choose occurrence index.")
            while True:
                raw_occurrence = _ask(input_fn, output_fn, "Occurrence index: ")
                try:
                    selected = int(raw_occurrence)
                except ValueError:
                    output_fn("Please enter a non-negative occurrence index.")
                    continue
                if 0 <= selected < len(matches):
                    occurrence = selected
                    break
                output_fn("Occurrence index is out of range.")
        try:
            resolve_exact_span(canonical_text, source, occurrence=occurrence)
        except ValueError as error:
            output_fn(f"Span rejected: {error}")
            continue
        return ProposedSpan(text=source, occurrence=occurrence)


def _spans(
    canonical_text: str,
    input_fn: InputFunction,
    output_fn: OutputFunction,
    label: str,
) -> tuple[ProposedSpan, ...]:
    return tuple(
        _span(canonical_text, input_fn, output_fn, f"{label} {index + 1}")
        for index in range(_count(input_fn, output_fn, label))
    )


def _display_candidates(record: AnnotationRecord, output_fn: OutputFunction) -> None:
    grouped: dict[str, list[Candidate]] = {prefix: [] for prefix in _PREFIXES}
    for candidate in record.candidate_registry.candidates:
        grouped[candidate.candidate_id[0]].append(candidate)
    output_fn("Deterministic candidates:")
    for prefix in _PREFIXES:
        output_fn(f"  {prefix} ({_PREFIX_LABELS[prefix]}):")
        candidates = grouped[prefix]
        if not candidates:
            output_fn("    none")
            continue
        for candidate in candidates:
            normalized = candidate.normalized.normalized_value
            normalized_display = normalized if normalized is not None else "(none)"
            output_fn(
                f"    {candidate.candidate_id}: {candidate.span.text} "
                f"[{candidate.span.start_char}:{candidate.span.end_char}] "
                f"normalized={normalized_display}"
            )


def _display_existing(record: AnnotationRecord, output_fn: OutputFunction) -> None:
    if record.annotation_status != "in_review" or record.human_annotations is None:
        return
    output_fn("Existing in_review annotations:")
    output_fn(json.dumps(record.human_annotations.model_dump(mode="json"), ensure_ascii=False))


def _refs(
    record: AnnotationRecord,
    input_fn: InputFunction,
    output_fn: OutputFunction,
    label: str,
    prefix: str,
) -> tuple[str, ...]:
    allowed = {
        candidate.candidate_id
        for candidate in record.candidate_registry.candidates
        if candidate.candidate_id.startswith(prefix)
    }
    while True:
        raw = _ask(
            input_fn,
            output_fn,
            f"{label} candidate IDs, comma-separated [blank for none]: ",
        )
        if not raw:
            return ()
        values = tuple(item.strip() for item in raw.split(",") if item.strip())
        if len(set(values)) != len(values):
            output_fn("Candidate IDs must not be duplicated.")
            continue
        invalid = [value for value in values if value not in allowed]
        if invalid:
            output_fn(f"Invalid {prefix} candidate ID(s): {', '.join(invalid)}")
            continue
        return values


def _modality(input_fn: InputFunction, output_fn: OutputFunction) -> Modality:
    while True:
        value = _ask(
            input_fn,
            output_fn,
            "Rule modality (obligation/prohibition/permission): ",
        )
        try:
            return Modality(value)
        except ValueError:
            output_fn("Choose obligation, prohibition, or permission.")


def _rule(
    record: AnnotationRecord,
    input_fn: InputFunction,
    output_fn: OutputFunction,
) -> ProposedRule:
    modality = _modality(input_fn, output_fn)
    actor = (
        _span(record.canonical_text, input_fn, output_fn, "actor")
        if _yes_no(input_fn, output_fn, "Does this rule have a literal actor?")
        else None
    )
    return ProposedRule(
        modality=modality,
        actor=actor,
        action=_span(record.canonical_text, input_fn, output_fn, "action"),
        conditions=_spans(record.canonical_text, input_fn, output_fn, "conditions"),
        exceptions=_spans(record.canonical_text, input_fn, output_fn, "rule exceptions"),
        deadline_refs=_refs(record, input_fn, output_fn, "Deadline", "T"),
        effective_date_refs=_refs(record, input_fn, output_fn, "Effective-date", "T"),
        monetary_threshold_refs=_refs(record, input_fn, output_fn, "Monetary-threshold", "M"),
        percentage_threshold_refs=_refs(record, input_fn, output_fn, "Percentage-threshold", "P"),
    )


def _proposal(
    record: AnnotationRecord,
    input_fn: InputFunction,
    output_fn: OutputFunction,
) -> SemanticProposal:
    return SemanticProposal(
        schema_version="phase11-proposal-v1",
        regulated_entities=_spans(record.canonical_text, input_fn, output_fn, "regulated entities"),
        rules=tuple(
            _rule(record, input_fn, output_fn)
            for _ in range(_count(input_fn, output_fn, "normative rules"))
        ),
        exceptions=_spans(record.canonical_text, input_fn, output_fn, "top-level exceptions"),
        penalties=_spans(record.canonical_text, input_fn, output_fn, "penalties"),
        deadline_refs=_refs(record, input_fn, output_fn, "Top-level deadline", "T"),
        effective_date_refs=_refs(record, input_fn, output_fn, "Top-level effective-date", "T"),
        monetary_threshold_refs=_refs(
            record, input_fn, output_fn, "Top-level monetary-threshold", "M"
        ),
        percentage_threshold_refs=_refs(
            record, input_fn, output_fn, "Top-level percentage-threshold", "P"
        ),
    )


def _update(
    proposal: SemanticProposal,
    input_fn: InputFunction,
    output_fn: OutputFunction,
) -> AnnotationUpdate:
    if _yes_no(
        input_fn,
        output_fn,
        "Save as reviewed with human_verified=true?",
    ):
        return AnnotationUpdate(
            human_annotations=proposal,
            annotation_status="reviewed",
            human_verified=True,
        )
    if _yes_no(input_fn, output_fn, "Save unfinished as in_review?"):
        return AnnotationUpdate(
            human_annotations=proposal,
            annotation_status="in_review",
            human_verified=False,
        )
    raise InteractiveAnnotationError("annotation was not saved")


def run_interactive_dev_annotation(
    *,
    input_fn: InputFunction = _terminal_input,
    output_fn: OutputFunction = print,
    next_context: NextContext = next_dev_annotation_context,
    validate_update: ValidateUpdate = validate_dev_annotation_update,
    save_update: SaveUpdate = save_dev_annotation_update,
    progress_fn: ProgressFunction = annotation_progress,
) -> dict[str, object]:
    """Annotate exactly one next-unreviewed DEV record in the terminal."""

    record, position, total = next_context()
    output_fn(f"Record {position}/{total}")
    output_fn(f"canonical_unit_id: {record.canonical_unit_id}")
    output_fn(f"canonical_text: {record.canonical_text}")
    _display_candidates(record, output_fn)
    _display_existing(record, output_fn)

    if _yes_no(
        input_fn, output_fn, "Does this clause contain any target Phase-11 semantic extraction?"
    ):
        proposal = _proposal(record, input_fn, output_fn)
    else:
        output_fn("Keeping the deterministic candidate registry unchanged.")
        proposal = SemanticProposal(schema_version="phase11-proposal-v1")

    preliminary = AnnotationUpdate(
        human_annotations=proposal,
        annotation_status="in_review",
        human_verified=False,
    )
    errors = validate_update(record.canonical_unit_id, preliminary)
    if errors:
        output_fn("Validation errors; nothing was saved:")
        for error in errors:
            output_fn(f"- {error}")
        raise InteractiveAnnotationError("interactive annotation failed validation")

    final_update = _update(proposal, input_fn, output_fn)
    errors = validate_update(record.canonical_unit_id, final_update)
    if errors:
        output_fn("Validation errors; nothing was saved:")
        for error in errors:
            output_fn(f"- {error}")
        raise InteractiveAnnotationError("interactive annotation failed validation")
    result = save_update(record.canonical_unit_id, final_update)
    output_fn(json.dumps({"progress": progress_fn("dev")}, ensure_ascii=False, sort_keys=True))
    output_fn("Next: uv run kawaneen extraction annotate-dev --next --interactive")
    return result
