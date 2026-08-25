from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.contracts import ExactSourceSpan
from kawaneen.extraction.deterministic import run_deterministic
from kawaneen.extraction.evaluation import evaluate_extractions, score_spans


def span(text: str, start: int) -> ExactSourceSpan:
    return ExactSourceSpan(
        text=text,
        start_char=start,
        end_char=start + len(text),
        canonical_unit_id="u1",
        document_id="d1",
    )


def test_span_metrics_report_tp_fp_fn_and_zero_support() -> None:
    metrics = score_spans((span("أ", 0),), (span("أ", 0), span("ب", 2)))
    assert metrics == {
        "TP": 1,
        "FP": 1,
        "FN": 0,
        "support": 1,
        "precision": 0.5,
        "recall": 1.0,
        "F1": 2 / 3,
    }
    assert score_spans((), ()) == {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "support": 0,
        "precision": 0.0,
        "recall": 0.0,
        "F1": 0.0,
    }


def test_evaluation_reports_strict_fields_rule_metrics_micro_macro_and_errors() -> None:
    provenance = SourceProvenance(
        source_id="saudi-moj-derived",
        source_version="8",
        source_path="local",
        source_row=1,
        source_field="text",
    )
    gold = run_deterministic(
        "المادة (7)", canonical_unit_id="u1", document_id="d1", source_provenance=provenance
    )
    predicted = run_deterministic(
        "المادة (8)", canonical_unit_id="u1", document_id="d1", source_provenance=provenance
    )
    report = evaluate_extractions(gold, predicted)
    assert report["field_metrics"]["referenced_articles"]["FN"] == 1
    assert report["micro"]["F1"] == 0.0
    assert "macro_F1" in report["aggregate"]
    assert report["clause_exact_match_accuracy"] == 1.0
    assert report["rule_metrics"]["modality_accuracy"] == 0.0
    assert "SPAN_BOUNDARY_ERROR" in report["error_counts"]
    assert report["engineering_metrics"]["FinalSchemaValidityRate"] == 1.0
