from __future__ import annotations

import json
from pathlib import Path

from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.evaluation.candidates import discover_evidence
from kawaneen.evaluation.models import QueryCategory


def _unit(text: str, unit_type: UnitType = UnitType.COURT_REASONING) -> CanonicalUnit:
    return CanonicalUnit(
        unit_id="unit-fixture",
        document_id="doc-fixture",
        unit_type=unit_type,
        text=text,
        provenance=SourceProvenance(
            source_id="alarb",
            source_version="fixture",
            source_path="fixture",
            source_row=1,
            source_field=unit_type.value,
        ),
    )


def test_evidence_discovery_rejects_punctuation_and_list_prefixes() -> None:
    assert discover_evidence(QueryCategory.EXACT_PROVISION, _unit("['.']")) == ()
    assert discover_evidence(QueryCategory.EXACT_PROVISION, _unit("['1. ح']")) == ()


def test_evidence_discovery_rejects_facts_as_exact_provision() -> None:
    unit = _unit("المادة 16 وردت في أقوال المدعي فقط.", UnitType.FACTS)
    assert discover_evidence(QueryCategory.EXACT_PROVISION, unit) == ()


def test_evidence_discovery_requires_real_semantic_case_holding() -> None:
    assert discover_evidence(QueryCategory.CASE_HOLDING, _unit("تتلخص وقائع الدعوى.")) == ()
    assert discover_evidence(
        QueryCategory.CASE_HOLDING,
        _unit(
            "قضت الدائرة التجارية برفض الدعوى وإلزام المدعى عليه بالمصاريف.",
            UnitType.VERDICT,
        ),
    )


def test_evidence_discovery_finds_non_prefix_exact_span() -> None:
    unit = _unit("تمهيد عام. واستندت المحكمة إلى المادة 16 من النظام.")
    spans = discover_evidence(QueryCategory.EXACT_PROVISION, unit)
    assert spans
    assert spans[0].start > 0
    assert "المادة 16" in unit.text[spans[0].start : spans[0].end]


def test_list_evidence_span_excludes_structural_prefix() -> None:
    unit = _unit(
        "['1. واستندت المحكمة إلى المادة 16 من النظام.']",
        UnitType.APPLICABLE_LAWS,
    )
    spans = discover_evidence(QueryCategory.EXACT_PROVISION, unit)
    assert spans
    assert unit.text[spans[0].start : spans[0].end].startswith("واستندت")
    assert not unit.text[spans[0].start : spans[0].end].startswith("1.")


def test_v2_draft_uses_natural_queries_and_non_copied_answers(tmp_path: Path) -> None:
    from kawaneen.evaluation.candidates import build_draft_candidates
    from kawaneen.evaluation.corpus import freeze_evaluation_corpus, load_evaluation_units

    corpus = freeze_evaluation_corpus(
        load_evaluation_units(Path("data/interim/canonical")),
        canonical_root=Path("data/interim/canonical"),
    )
    result = build_draft_candidates(corpus, output_root=tmp_path)
    assert len(result.base_candidates) >= 320
    assert len(result.selected_base_candidates) == 200
    assert all("intent-" not in item.query_text for item in result.all_items)
    assert all("Internal reference" not in item.query_text for item in result.all_items)
    assert all("مرجع داخلي" not in item.query_text for item in result.all_items)
    assert all(
        item.answerability.value == "unanswerable" or item.gold_answer not in item.query_text
        for item in result.all_items
    )
    assert len(result.variants) == 40
    assert {item.variant_id for item in result.variants} == {
        "simple-ar",
        "egyptian-ar",
        "english",
        "code-switch",
    }
    assert all(
        item.query_text
        not in {
            "هل نقدر نجاوب السؤال ده من الأحكام المتاحة؟",
            "ينفع نعرف الإجابة دي من المستندات الموجودة؟",
            "Can this question be answered from the available judgments?",
            "هل نقدر نحدد الـ answer من الأحكام المتاحة؟",
        }
        for item in result.variants
    )
    base_by_intent = {item.intent_id: item for item in result.selected_base_candidates}
    for variant in result.variants:
        parent = base_by_intent[variant.base_intent_id or ""]
        assert variant.evidence_groups == parent.evidence_groups
        assert variant.gold_answer == parent.gold_answer
        assert variant.answerability == parent.answerability

    unanswerables = [
        item
        for item in result.selected_base_candidates
        if item.category is QueryCategory.UNANSWERABLE
    ]
    assert len({item.query_text for item in unanswerables}) == len(unanswerables)
    assert all(item.gold_answer is None and not item.evidence_groups for item in unanswerables)

    from kawaneen.evaluation.diagnostics import build_review_diagnostics

    diagnostics = build_review_diagnostics(result.all_items, corpus.units)
    assert len(diagnostics) == 240
    assert all(record["machine_quality"]["pass"] is True for record in diagnostics)


def test_compact_handoff_preserves_display_text(tmp_path: Path) -> None:
    from kawaneen.evaluation.corpus import freeze_evaluation_corpus
    from kawaneen.evaluation.handoff import write_handoff_artifacts

    unit = _unit("display text: ؛ [exact]", UnitType.VERDICT)
    corpus = freeze_evaluation_corpus(
        (unit,),
        canonical_root=Path("data/interim/canonical"),
    )
    result = write_handoff_artifacts(corpus, (), tmp_path)
    shard_path = tmp_path / "canonical_review_shards" / result["shards"][0]["name"]
    row = json.loads(shard_path.read_text(encoding="utf-8"))
    assert row["display_text"] == unit.text
    manifest = json.loads((tmp_path / "canonical_review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["included_fields"] == [
        "source_id",
        "source_version",
        "document_id",
        "unit_id",
        "unit_type",
        "display_text",
    ]
