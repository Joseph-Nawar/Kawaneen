from kawaneen.evaluation.models import Answerability, QueryLanguage, QueryRegister
from kawaneen.retrieval.analysis import complementarity_top10, robustness_parent_variant


def test_complementarity_counts_four_outcomes() -> None:
    left = {
        "q1": {"Recall@10": 1.0},
        "q2": {"Recall@10": 1.0},
        "q3": {"Recall@10": 0.0},
        "q4": {"Recall@10": 0.0},
    }
    right = {
        "q1": {"Recall@10": 1.0},
        "q2": {"Recall@10": 0.0},
        "q3": {"Recall@10": 1.0},
        "q4": {"Recall@10": 0.0},
    }

    assert complementarity_top10(left, right) == {
        "lexical_succeeds_dense_fails": 1,
        "dense_succeeds_lexical_fails": 1,
        "both_succeed": 1,
        "both_fail": 1,
        "sample_count": 4,
    }


def test_robustness_groups_all_supported_variant_families() -> None:
    def item(
        query_id: str,
        *,
        variant_id: str | None = None,
        language=QueryLanguage.ARABIC,
        register=QueryRegister.FORMAL,
    ):
        return type(
            "Item",
            (),
            {
                "intent_id": "intent-0",
                "base_intent_id": "intent-0",
                "query_id": query_id,
                "variant_id": variant_id,
                "answerability": Answerability.ANSWERABLE,
                "language": language,
                "register": register,
            },
        )()

    base = item("base")
    variants = (
        item("simple", variant_id="simple"),
        item("egyptian", variant_id="egyptian", register=QueryRegister.EGYPTIAN),
        item("english", variant_id="english", language=QueryLanguage.ENGLISH),
        item("code", variant_id="code", language=QueryLanguage.CODE_SWITCHED),
    )
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
    rows = {
        query_id: {metric: float(index) for metric in metrics}
        for index, query_id in enumerate(("base", "simple", "egyptian", "english", "code"))
    }

    result = robustness_parent_variant((base, *variants), rows)

    assert set(result) == {
        "simple_arabic",
        "egyptian_arabic",
        "english",
        "arabic_english_code_switch",
    }
    assert all(value["sample_count"] == 1 for value in result.values())
