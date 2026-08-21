from __future__ import annotations

import pytest
from test_grounding_assembly import FakeCounter, inputs, resolver

from kawaneen.cli import build_parser
from kawaneen.evaluation.models import (
    Answerability,
    ChunkQrel,
    CreationMethod,
    DatasetItem,
    Difficulty,
    EvidenceGroup,
    EvidenceSpan,
    QueryCategory,
    QueryLanguage,
    QueryRegister,
    QueryType,
    RelevanceGrade,
)
from kawaneen.grounding.artifacts import context_pack_fingerprint, write_tracked_json
from kawaneen.grounding.assembly import ContextAssembler
from kawaneen.grounding.evaluation import audit_dev_contexts, audit_evidence_retention


def test_audit_reports_overlap_deduplication_and_zero_invariant_violations(tmp_path) -> None:
    corpus = resolver(tmp_path)
    pack = ContextAssembler(corpus, FakeCounter(), max_context_tokens=1000).assemble(
        query_id="q1",
        ranked_inputs=inputs("c-a23", "c-a2", "c-a1"),
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
    )
    metrics = audit_dev_contexts(
        (pack,),
        {"q1": inputs("c-a23", "c-a2", "c-a1")},
        resolver=corpus,
    )
    assert metrics["query_count"] == 1
    assert metrics["overlapping_or_repeated_units_removed"] == 1
    assert metrics["duplicate_unit_violations"] == 0
    assert metrics["ordering_violations"] == 0
    assert metrics["dedup_only_representation_losses"] == 0
    assert metrics["token_budget_violations"] == 0
    assert metrics["mid_unit_truncations"] == 0


def test_context_pack_fingerprint_is_stable_and_input_sensitive(tmp_path) -> None:
    corpus = resolver(tmp_path)
    pack = ContextAssembler(corpus, FakeCounter(), max_context_tokens=1000).assemble(
        query_id="q1",
        ranked_inputs=inputs("c-a1"),
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
    )
    kwargs = {
        "phase8_selection_sha256": "a" * 64,
        "query_id": "q1",
        "canonical_corpus_hash": "b" * 64,
        "assembly_policy_version": "phase9-context-assembly-v1",
        "token_counter": FakeCounter(),
        "max_context_tokens": 1000,
    }
    assert context_pack_fingerprint(pack, **kwargs) == context_pack_fingerprint(pack, **kwargs)
    changed = pack.model_copy(update={"input_chunk_ids": ("different",)})
    assert context_pack_fingerprint(pack, **kwargs) != context_pack_fingerprint(changed, **kwargs)


def test_audit_separates_input_coverage_from_assembly_conditional_retention(tmp_path) -> None:
    corpus = resolver(tmp_path)
    ranked = inputs("c-a1")
    pack = ContextAssembler(corpus, FakeCounter(), max_context_tokens=0).assemble(
        query_id="q1",
        ranked_inputs=ranked,
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
    )
    item = DatasetItem(
        query_id="q1",
        intent_id="i1",
        query_text="question",
        language=QueryLanguage.ARABIC,
        register=QueryRegister.FORMAL,
        category=QueryCategory.EXACT_PROVISION,
        query_type=QueryType.REFERENCE_LOOKUP,
        jurisdiction="test",
        creation_method=CreationMethod.DOCUMENT_DERIVED,
        answerability=Answerability.ANSWERABLE,
        difficulty=Difficulty.EASY,
        evidence_groups=(
            EvidenceGroup(
                group_id="g1",
                spans=(EvidenceSpan(unit_id="a1", start=0, end=2, grade=RelevanceGrade.REQUIRED),),
            ),
        ),
        chunk_qrels=(ChunkQrel(chunk_id="c-a1", grade=RelevanceGrade.REQUIRED),),
        gold_answer="answer",
    )
    metrics = audit_dev_contexts(
        (pack,),
        {"q1": ranked},
        resolver=corpus,
        items=(item,),
    )
    assert metrics["InputGoldEvidenceCoverage@8"] == {"hits": 1, "queries": 1, "rate": 1.0}
    assert metrics["InputCompleteGoldEvidenceCoverage@8"] == {
        "hits": 1,
        "queries": 1,
        "rate": 1.0,
    }
    assert metrics["AssemblyConditionalGoldRetention"] == {
        "retained_queries": 0,
        "input_covered_queries": 1,
        "assembly_representation_losses": 1,
        "conditional_rate": 0.0,
    }
    assert metrics["AssemblyConditionalCompleteGoldRetention"] == {
        "retained_queries": 0,
        "input_covered_queries": 1,
        "assembly_representation_losses": 1,
        "conditional_rate": 0.0,
    }
    unbounded = ContextAssembler(corpus, FakeCounter(), max_context_tokens=1000).assemble(
        query_id="q1",
        ranked_inputs=ranked,
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
    )
    retention = audit_evidence_retention(
        (pack,),
        (unbounded,),
        {"q1": ranked},
        resolver=corpus,
        items=(item,),
    )
    assert retention["UnboundedConditionalGoldRetention"] == {
        "retained_queries": 1,
        "input_covered_queries": 1,
        "assembly_representation_losses": 0,
        "conditional_rate": 1.0,
    }
    assert retention["UnboundedConditionalCompleteGoldRetention"] == {
        "retained_queries": 1,
        "input_covered_queries": 1,
        "assembly_representation_losses": 0,
        "conditional_rate": 1.0,
    }
    assert retention["loss_attribution"] == {
        "canonical_unit_deduplication": 0,
        "provenance_resolution_failure": 0,
        "block_reconstruction": 0,
        "ordering": 0,
        "token_budget_exclusion": 1,
        "other": 0,
    }
    assert retention["BudgetOnlyLosses"]["gold_representation_losses"] == 1


def test_tracked_writer_rejects_source_text_but_cli_has_grounding_commands(tmp_path) -> None:
    with pytest.raises(ValueError, match="source text"):
        write_tracked_json(tmp_path / "bad.json", {"display_text": "secret"})
    assert (
        build_parser().parse_args(["grounding", "assemble-dev"]).grounding_command
        == "assemble-dev"
    )
    assert build_parser().parse_args(["grounding", "audit-dev"]).grounding_command == "audit-dev"
