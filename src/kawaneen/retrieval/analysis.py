"""Quantitative complementarity and robustness summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kawaneen.evaluation.models import Answerability, DatasetItem


def complementarity_top10(
    lexical: Mapping[str, Mapping[str, float]], dense: Mapping[str, Mapping[str, float]]
) -> dict[str, int]:
    query_ids = sorted(set(lexical) & set(dense))
    outcomes = {
        "lexical_succeeds_dense_fails": 0,
        "dense_succeeds_lexical_fails": 0,
        "both_succeed": 0,
        "both_fail": 0,
    }
    for query_id in query_ids:
        lexical_success = lexical[query_id]["Recall@10"] > 0
        dense_success = dense[query_id]["Recall@10"] > 0
        if lexical_success and dense_success:
            outcomes["both_succeed"] += 1
        elif lexical_success:
            outcomes["lexical_succeeds_dense_fails"] += 1
        elif dense_success:
            outcomes["dense_succeeds_lexical_fails"] += 1
        else:
            outcomes["both_fail"] += 1
    return {**outcomes, "sample_count": len(query_ids)}


def robustness_parent_variant(
    items: Sequence[DatasetItem], rows: Mapping[str, Mapping[str, float]]
) -> dict[str, object]:
    """Compare each answerable robustness variant with its base intent."""
    bases = {
        item.intent_id: item
        for item in items
        if item.variant_id is None and item.answerability is Answerability.ANSWERABLE
    }
    metrics = (
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "nDCG@10",
        "Precision@5",
        "CompleteEvidenceRecall@5",
        "CompleteEvidenceRecall@10",
    )
    grouped: dict[str, list[dict[str, float]]] = {}
    for variant in items:
        if (
            variant.variant_id is None
            or variant.answerability is not Answerability.ANSWERABLE
            or variant.base_intent_id not in bases
            or variant.query_id not in rows
            or bases[variant.base_intent_id].query_id not in rows
        ):
            continue
        if variant.language.value == "en":
            label = "english"
        elif variant.language.value == "ar-en":
            label = "arabic_english_code_switch"
        elif variant.register.value == "egyptian":
            label = "egyptian_arabic"
        else:
            label = "simple_arabic"
        parent = rows[bases[variant.base_intent_id].query_id]
        child = rows[variant.query_id]
        grouped.setdefault(label, []).append(
            {metric: parent[metric] - child[metric] for metric in metrics}
        )
    output: dict[str, object] = {}
    for label, values in sorted(grouped.items()):
        output[label] = {
            "sample_count": len(values),
            "mean_parent_minus_variant": {
                metric: sum(value[metric] for value in values) / len(values) for metric in metrics
            },
        }
    return output
