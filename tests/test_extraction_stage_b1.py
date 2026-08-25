from __future__ import annotations

from pathlib import Path

from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.annotation import AnnotationRecord
from kawaneen.extraction.contracts import CandidateRegistry
from kawaneen.extraction.deterministic import run_deterministic
from kawaneen.extraction.hybrid import assemble_hybrid_result
from kawaneen.extraction.hybrid_prompt import render_hybrid_prompt
from kawaneen.extraction.hybrid_runtime import run_hybrid_records
from kawaneen.extraction.orchestration import (
    HYBRID_STAGE_B1_CLEAN_CHECKPOINT_ROOT,
    HYBRID_STAGE_B1_CLEAN_RESULT_ROOT,
    HYBRID_STAGE_B2_CLEAN_CHECKPOINT_ROOT,
    HYBRID_STAGE_B2_CLEAN_RESULT_ROOT,
)


def _base(text: str):
    return run_deterministic(
        text,
        canonical_unit_id="u1",
        document_id="d1",
        source_provenance=SourceProvenance(
            source_id="saudi-moj-derived",
            source_version="v3",
            source_path="private",
            source_row=1,
            source_field="text",
        ),
    )


def _registry(text: str) -> CandidateRegistry:
    registry = _base(text).candidate_registry
    assert registry is not None
    return registry


def _proposal(**updates: object) -> dict[str, object]:
    proposal: dict[str, object] = {
        "schema_version": "phase11-proposal-v1",
        "regulated_entities": [],
        "rules": [],
        "exceptions": [],
        "penalties": [],
        "deadline_refs": [],
        "effective_date_refs": [],
        "monetary_threshold_refs": [],
        "percentage_threshold_refs": [],
    }
    proposal.update(updates)
    return proposal


def test_wrong_type_and_empty_candidate_ids_are_dropped_but_valid_id_remains() -> None:
    text = "يجب التسجيل خلال 30 يوماً."
    result = assemble_hybrid_result(
        text,
        _base(text),
        _proposal(
            rules=[
                {
                    "modality": "obligation",
                    "action": {"text": "التسجيل"},
                    "deadline_refs": ["T001", ""],
                    "percentage_threshold_refs": ["T001", ""],
                }
            ]
        ),
    )
    assert result.rules[0].deadline_refs == ("T001",)
    assert result.rules[0].percentage_threshold_refs == ()
    assert all(
        "rejected_id=" in diagnostic.message
        for diagnostic in result.validation_metadata.diagnostics
        if diagnostic.code == "INVALID_CANDIDATE_REFERENCE"
    )


def test_unique_exact_span_with_bad_occurrence_is_resolved_and_diagnosed() -> None:
    text = "يجوز التمديد."
    result = assemble_hybrid_result(
        text,
        _base(text),
        _proposal(
            rules=[
                {
                    "modality": "permission",
                    "action": {"text": "التمديد", "occurrence": 9},
                }
            ]
        ),
    )
    assert result.permissions[0].action_span.text == "التمديد"
    assert any(
        diagnostic.code == "INVALID_OCCURRENCE_CORRECTED"
        for diagnostic in result.validation_metadata.diagnostics
    )


def test_repeated_exact_span_with_invalid_occurrence_is_rejected() -> None:
    text = "يجوز التمديد ويجوز التمديد."
    result = assemble_hybrid_result(
        text,
        _base(text),
        _proposal(
            rules=[
                {
                    "modality": "permission",
                    "action": {"text": "التمديد", "occurrence": 0},
                    "conditions": [{"text": "يجوز", "occurrence": 9}],
                }
            ]
        ),
    )
    assert result.permissions[0].condition_spans == ()
    assert any(
        diagnostic.code == "AMBIGUOUS_OR_INVALID_OCCURRENCE"
        for diagnostic in result.validation_metadata.diagnostics
    )


def test_invalid_action_drops_only_its_rule_and_all_invalid_is_valid_empty_result() -> None:
    text = "يجوز التمديد."
    result = assemble_hybrid_result(
        text,
        _base(text),
        _proposal(
            regulated_entities=[{"text": "غير موجود"}],
            rules=[
                {"modality": "permission", "action": {"text": "غير موجود"}},
            ],
            penalties=[{"text": "غير موجود"}],
        ),
    )
    assert result.rules == ()
    assert result.regulated_entities == ()
    assert result.penalties == ()
    assert result.validation_metadata.diagnostics


def test_valid_rules_are_assembled_in_the_contract_partition_order() -> None:
    text = "يجوز التمديد ولا يجوز التأخير ويجب التسجيل."
    result = assemble_hybrid_result(
        text,
        _base(text),
        _proposal(
            rules=[
                {"modality": "prohibition", "action": {"text": "التأخير"}},
                {"modality": "obligation", "action": {"text": "التسجيل"}},
                {"modality": "permission", "action": {"text": "التمديد"}},
            ]
        ),
    )
    assert [rule.modality.value for rule in result.rules] == [
        "obligation",
        "prohibition",
        "permission",
    ]


def test_prompt_contains_typed_candidate_allowlists() -> None:
    text = "المادة (7) ويجب التسجيل خلال 30 يوماً."
    rendered = render_hybrid_prompt(text, _registry(text))
    assert "VALID_TYPED_CANDIDATE_ALLOWLISTS" in rendered.text
    assert "deadline_refs" in rendered.text
    assert "never emit empty-string IDs" in rendered.text
    assert "T001" in rendered.text


def test_stage_b2_prompt_contains_only_synthetic_guidance() -> None:
    from kawaneen.extraction.hybrid_prompt import (
        HYBRID_STAGE_B2_PROMPT_TEMPLATE_VERSION,
        render_hybrid_prompt,
    )

    text = "يجب على الجهة تقديم التقرير."
    rendered = render_hybrid_prompt(
        text, _registry(text), HYBRID_STAGE_B2_PROMPT_TEMPLATE_VERSION
    )
    assert "يجب على المرخص له تقديم التقرير إلى الهيئة." in rendered.text
    assert "يحظر على المنشأة إفشاء المعلومات السرية." in rendered.text
    assert "يجوز للجهة تمديد المهلة." in rendered.text
    assert "regulated_entities" in rendered.text
    assert "complete legal action" in rendered.text
    assert "distinct normative rules" in rendered.text
    assert "return []" in rendered.text
    assert "044c940e-26ec-5af0-8834-36f2dc7591c5" not in rendered.text
    assert "CANONICAL_TEXT:\n" + text in rendered.text


def test_b2_prompt_changes_do_not_change_schema() -> None:
    from kawaneen.extraction.hybrid_prompt import (
        HYBRID_STAGE_B2_PROMPT_TEMPLATE_VERSION,
        hybrid_prompt_hash,
        hybrid_schema_hash,
    )

    assert len(hybrid_prompt_hash(HYBRID_STAGE_B2_PROMPT_TEMPLATE_VERSION)) == 64
    assert hybrid_prompt_hash() != hybrid_prompt_hash(HYBRID_STAGE_B2_PROMPT_TEMPLATE_VERSION)
    assert hybrid_schema_hash() == hybrid_schema_hash()


def test_stage_b1_clean_namespace_is_separate_from_prior_artifacts() -> None:
    assert HYBRID_STAGE_B1_CLEAN_RESULT_ROOT.as_posix().endswith(
        "hybrid-qwen-v1-stage-b1-clean/dev"
    )
    assert HYBRID_STAGE_B1_CLEAN_CHECKPOINT_ROOT.as_posix().endswith(
        "hybrid-qwen-v1-stage-b1-clean/dev"
    )


def test_stage_b2_clean_namespace_is_separate_from_all_prior_artifacts() -> None:
    assert HYBRID_STAGE_B2_CLEAN_RESULT_ROOT.as_posix().endswith(
        "hybrid-qwen-v1-stage-b2-clean/dev"
    )
    assert HYBRID_STAGE_B2_CLEAN_CHECKPOINT_ROOT.as_posix().endswith(
        "hybrid-qwen-v1-stage-b2-clean/dev"
    )
    assert HYBRID_STAGE_B2_CLEAN_RESULT_ROOT != HYBRID_STAGE_B1_CLEAN_RESULT_ROOT
    assert HYBRID_STAGE_B2_CLEAN_CHECKPOINT_ROOT != HYBRID_STAGE_B1_CLEAN_CHECKPOINT_ROOT


class _TimeoutThenValidProvider:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0

    def propose(self, canonical_text: str, registry: object) -> object:
        del canonical_text, registry
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("timed out")
        return self.response


def _record(record_id: str) -> AnnotationRecord:
    text = "يجوز التمديد."
    result = _base(text)
    return AnnotationRecord(
        canonical_unit_id=record_id,
        document_id="d1",
        canonical_text=text,
        source_provenance=result.source_provenance,
        source_fingerprint="a" * 64,
        split="dev",
        candidate_registry=_registry(text),
    )


def test_timeout_is_counted_as_attempted_without_raw_response(tmp_path: Path) -> None:
    provider = _TimeoutThenValidProvider(
        _proposal(rules=[{"modality": "permission", "action": {"text": "التمديد"}}])
    )
    result = run_hybrid_records(
        [_record("u1"), _record("u2")],
        provider,  # type: ignore[arg-type]
        checkpoint_root=tmp_path / "checkpoints",
        result_root=tmp_path / "results",
        selection_fingerprint="b" * 64,
        semantic_release_fingerprint="c" * 64,
        candidate_compatible_release_fingerprint="d" * 64,
        prompt_hash="e" * 64,
        schema_hash="f" * 64,
        qwen_model="qwen",
        qwen_digest="sha256:" + "0" * 64,
        tokenizer_revision="a" * 40,
        accept_field_local_diagnostics=True,
    )
    assert result["provider_calls_attempted"] == 2
    assert result["raw_responses_received"] == 1
    assert result["timeouts"] == 1
