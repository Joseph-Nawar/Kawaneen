from __future__ import annotations

from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.annotation import AnnotationRecord, validate_annotation_record
from kawaneen.extraction.candidates import build_candidate_registry
from kawaneen.extraction.contracts import ProposedRule, ProposedSpan, SemanticProposal

TEXT = "الجهة تلتزم بالفعل خلال 30 يوماً إلا عند الشرط."


def _span(text: str, occurrence: int | None = None) -> ProposedSpan:
    return ProposedSpan(text=text, occurrence=occurrence)


def _record(proposal: SemanticProposal) -> AnnotationRecord:
    return AnnotationRecord(
        canonical_unit_id="synthetic-dev",
        document_id="synthetic-document",
        canonical_text=TEXT,
        source_provenance=SourceProvenance(
            source_id="saudi-moj-derived",
            source_version="synthetic",
            source_path="synthetic",
            source_row=1,
            source_field="text",
        ),
        source_fingerprint="f" * 64,
        split="dev",
        candidate_registry=build_candidate_registry(
            TEXT,
            canonical_unit_id="synthetic-dev",
            document_id="synthetic-document",
        ),
        human_annotations=proposal,
    )


def _rule(action: str, *, modality: str = "obligation") -> ProposedRule:
    return ProposedRule(
        modality=modality,
        actor=_span("الجهة"),
        action=_span(action),
        conditions=(_span("عند الشرط"),),
        exceptions=(_span("إلا"),),
        deadline_refs=("T001",),
    )


def test_cross_role_and_cross_rule_reuse_is_allowed() -> None:
    first = _rule("بالفعل")
    second = _rule("بالفعل", modality="permission")
    proposal = SemanticProposal(
        schema_version="phase11-proposal-v1",
        regulated_entities=(_span("الجهة"),),
        rules=(first, second),
        exceptions=(_span("إلا"),),
        deadline_refs=("T001",),
    )
    assert validate_annotation_record(_record(proposal), {"synthetic-dev"}) == []


def test_duplicate_homogeneous_collections_are_rejected() -> None:
    cases = [
        SemanticProposal(
            schema_version="phase11-proposal-v1",
            regulated_entities=(_span("الجهة"), _span("الجهة")),
        ),
        SemanticProposal(
            schema_version="phase11-proposal-v1",
            exceptions=(_span("إلا"), _span("إلا")),
        ),
        SemanticProposal(
            schema_version="phase11-proposal-v1",
            penalties=(_span("الشرط"), _span("الشرط")),
        ),
        SemanticProposal(
            schema_version="phase11-proposal-v1",
            rules=(
                ProposedRule(
                    modality="obligation",
                    action=_span("بالفعل"),
                    conditions=(_span("عند الشرط"), _span("عند الشرط")),
                ),
            ),
        ),
        SemanticProposal(
            schema_version="phase11-proposal-v1",
            rules=(_rule("بالفعل"),),
            deadline_refs=("T001", "T001"),
        ),
    ]
    for proposal in cases:
        errors = validate_annotation_record(_record(proposal), {"synthetic-dev"})
        assert any("duplicate" in error for error in errors)


def test_exact_duplicate_normative_rule_is_rejected() -> None:
    rule = _rule("بالفعل")
    proposal = SemanticProposal(schema_version="phase11-proposal-v1", rules=(rule, rule))
    errors = validate_annotation_record(_record(proposal), {"synthetic-dev"})
    assert any("duplicate" in error for error in errors)
