from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction import orchestration
from kawaneen.extraction.annotation import (
    AnnotationRecord,
    AnnotationUpdate,
    validate_annotation_record,
)
from kawaneen.extraction.candidates import build_candidate_registry
from kawaneen.extraction.contracts import SemanticProposal
from kawaneen.extraction.interactive import (
    InteractiveAnnotationError,
    run_interactive_dev_annotation,
)


class Answers:
    def __init__(self, values: list[str]) -> None:
        self._values: Iterator[str] = iter(values)

    def __call__(self, _prompt: str) -> str:
        try:
            return next(self._values)
        except StopIteration as error:
            raise AssertionError("interactive test requested an unexpected input") from error


def record(text: str) -> AnnotationRecord:
    return AnnotationRecord(
        canonical_unit_id="synthetic-dev-1",
        document_id="synthetic-document-1",
        canonical_text=text,
        source_provenance=SourceProvenance(
            source_id="saudi-moj-derived",
            source_version="synthetic",
            source_path="private",
            source_row=1,
            source_field="text",
        ),
        source_fingerprint="a" * 64,
        split="dev",
        candidate_registry=build_candidate_registry(
            text,
            canonical_unit_id="synthetic-dev-1",
            document_id="synthetic-document-1",
        ),
    )


def run(
    values: list[str],
    text: str,
    *,
    validator: Callable[[str, AnnotationUpdate], list[str]] | None = None,
    output: list[str] | None = None,
) -> tuple[list[AnnotationUpdate], list[str]]:
    current = record(text)
    saved: list[AnnotationUpdate] = []
    output = output if output is not None else []

    def save(_record_id: str, update: AnnotationUpdate) -> dict[str, object]:
        saved.append(update)
        return {"split": "dev", "annotation_status": update.annotation_status}

    result = run_interactive_dev_annotation(
        input_fn=Answers(values),
        output_fn=output.append,
        next_context=lambda: (current, 1, 1),
        validate_update=validator
        or (
            lambda _record_id, update: validate_annotation_record(
                current.model_copy(
                    update={
                        "human_annotations": update.human_annotations,
                        "annotation_status": update.annotation_status,
                        "human_verified": update.human_verified,
                    }
                ),
                {current.canonical_unit_id},
            )
        ),
        save_update=save,
        progress_fn=lambda _split: {
            "total": 1,
            "reviewed": int(bool(saved)),
            "human_verified": int(bool(saved and saved[0].human_verified)),
            "remaining": 0 if saved else 1,
            "invalid": 0,
        },
    )
    assert result["split"] == "dev"
    return saved, output


def test_positive_obligation_collects_exact_actor_and_action() -> None:
    saved, _ = run(
        [
            "y",
            "0",
            "1",
            "obligation",
            "y",
            "المرخص",
            "تقديم الطلب",
            "0",
            "0",
            "",
            "",
            "",
            "",
            "",
            "0",
            "0",
            "",
            "",
            "",
            "",
            "y",
        ],
        "يلتزم المرخص بتقديم الطلب.",
    )
    assert saved[0].human_annotations.rules[0].action.text == "تقديم الطلب"
    assert saved[0].human_annotations.rules[0].actor is not None


def test_negative_path_builds_empty_verified_annotation() -> None:
    saved, _ = run(["n", "y"], "يتناول النص تعريفاً فقط.")
    assert saved[0].human_annotations == SemanticProposal(schema_version="phase11-proposal-v1")
    assert saved[0].annotation_status == "reviewed"
    assert saved[0].human_verified is True


def test_in_review_annotations_are_displayed_for_the_next_record() -> None:
    current = record("يجب تقديم الطلب.").model_copy(
        update={
            "annotation_status": "in_review",
            "human_annotations": SemanticProposal(schema_version="phase11-proposal-v1"),
        }
    )
    output: list[str] = []
    saved: list[AnnotationUpdate] = []

    run_interactive_dev_annotation(
        input_fn=Answers(["n", "y"]),
        output_fn=output.append,
        next_context=lambda: (current, 1, 1),
        validate_update=lambda _id, _update: [],
        save_update=lambda _id, update: saved.append(update) or {"split": "dev"},
        progress_fn=lambda _split: {"total": 1},
    )

    assert any("Existing in_review annotations:" in line for line in output)
    assert saved[0].human_verified is True


def test_multiple_rules_and_candidate_reference_are_collected() -> None:
    saved, _ = run(
        [
            "y",
            "0",
            "2",
            "obligation",
            "n",
            "تقديم الطلب",
            "0",
            "0",
            "T001",
            "",
            "",
            "",
            "prohibition",
            "n",
            "إفشاء البيانات",
            "0",
            "0",
            "",
            "",
            "",
            "",
            "0",
            "0",
            "",
            "",
            "",
            "",
            "y",
        ],
        "يلتزم بتقديم الطلب خلال ٣٠ يوماً ويحظر إفشاء البيانات.",
    )
    proposal = saved[0].human_annotations
    assert len(proposal.rules) == 2
    assert proposal.rules[0].deadline_refs == ("T001",)
    assert proposal.rules[1].modality.value == "prohibition"


def test_invalid_exact_span_is_rejected_before_save() -> None:
    saved, output = [], []
    with pytest.raises(AssertionError):
        run(
            ["y", "0", "1", "obligation", "n", "غير موجود"],
            "يجب تقديم الطلب.",
            output=output,
        )
    assert saved == []
    assert any("not found" in line for line in output)


def test_duplicate_span_requests_occurrence_selection() -> None:
    saved, output = run(
        [
            "y",
            "0",
            "1",
            "obligation",
            "n",
            "الحفظ",
            "1",
            "0",
            "0",
            "",
            "",
            "",
            "",
            "0",
            "0",
            "",
            "",
            "",
            "",
            "y",
        ],
        "يلتزم بالحفظ والحفظ.",
    )
    action = saved[0].human_annotations.rules[0].action
    assert action.text == "الحفظ"
    assert action.occurrence == 1
    assert any("occurrence" in line for line in output)


def test_unfinished_record_is_saved_in_review() -> None:
    saved, _ = run(["n", "n", "y"], "نص وصفي.")
    assert saved[0].annotation_status == "in_review"
    assert saved[0].human_verified is False


def test_validator_errors_are_shown_and_verified_save_is_blocked() -> None:
    saved, output = [], []
    with pytest.raises(InteractiveAnnotationError):
        run(
            ["n", "y"],
            "نص وصفي.",
            validator=lambda _id, _update: ["synthetic validation error"],
            output=output,
        )
    assert saved == []
    assert any("synthetic validation error" in line for line in output)


def test_interactive_context_is_dev_only(monkeypatch: pytest.MonkeyPatch) -> None:
    current = record("نص وصفي.")
    calls: list[str] = []

    def load(split: str, **_kwargs: object) -> list[AnnotationRecord]:
        calls.append(split)
        assert split == "dev"
        return [current]

    monkeypatch.setattr(orchestration, "_load_records", load)
    selected, position, total = orchestration.next_dev_annotation_context()
    assert selected.canonical_unit_id == current.canonical_unit_id
    assert (position, total) == (1, 1)
    assert calls == ["dev"]


def test_interactive_does_not_invoke_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def provider_call(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("provider invocation is forbidden")

    monkeypatch.setattr(orchestration, "run_hybrid_split", provider_call)
    run(["n", "y"], "نص وصفي.")
    assert called is False
