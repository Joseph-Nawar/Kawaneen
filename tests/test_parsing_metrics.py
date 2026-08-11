from kawaneen.parsing.benchmark import (
    BenchmarkBlock,
    assign_blocks_to_gold_region,
    calculate_anchored_metrics,
    calculate_metrics,
    calculate_region_metrics,
    canonicalize_prediction_box,
)


def test_benchmark_metrics_are_deterministic_and_include_page_and_article_checks() -> None:
    metrics = calculate_metrics(
        reference="Article 1\nAlpha\nPage 2",
        hypothesis="Article 1\nAlphb\nPage 2",
        reference_headings=("Article 1",),
        hypothesis_headings=("Article 1",),
        reference_articles=("1",),
        hypothesis_articles=("1",),
        reference_pages=(2,),
        hypothesis_pages=(2,),
    )
    assert metrics.cer > 0
    assert metrics.wer > 0
    assert metrics.heading_f1 == 1
    assert metrics.semantic_article_number_accuracy == 1
    assert metrics.page_reference_preservation == 1


def test_word_error_rate_uses_word_operations_not_character_operations() -> None:
    metrics = calculate_metrics(
        reference="alpha beta gamma",
        hypothesis="alpha delta gamma",
        reference_headings=(),
        hypothesis_headings=(),
        reference_articles=(),
        hypothesis_articles=(),
        reference_pages=(),
        hypothesis_pages=(),
    )
    assert metrics.wer == 1 / 3


def test_region_metrics_match_by_identity_instead_of_page_concatenation() -> None:
    metrics = calculate_region_metrics(
        reference=(
            BenchmarkBlock("r1", "Article 1", "heading", (0, 0, 100, 10), 1),
            BenchmarkBlock("r2", "Alpha", "paragraph", (0, 20, 100, 30), 2),
        ),
        hypothesis=(
            BenchmarkBlock("r2", "Alpha", "paragraph", (0, 20, 100, 30), 1),
            BenchmarkBlock("r1", "Article 1", "heading", (0, 0, 100, 10), 2),
        ),
    )

    assert metrics.cer == 0
    assert metrics.wer == 0
    assert metrics.heading_f1 == 1
    assert metrics.reading_order_accuracy == 0


def test_region_metrics_does_not_match_nearby_but_unrelated_blocks() -> None:
    metrics = calculate_region_metrics(
        reference=(BenchmarkBlock("r1", "المادة الأولى", "article_label", (0, 0, 100, 10), 1),),
        hypothesis=(
            BenchmarkBlock("wrong", "المادة الثانية", "article_label", (0, 0, 100, 10), 1),
        ),
    )

    assert metrics.exact_article_number_accuracy == 0
    assert metrics.semantic_article_number_accuracy == 0


def test_region_metrics_uses_geometry_only_when_both_block_ids_are_absent() -> None:
    metrics = calculate_region_metrics(
        reference=(BenchmarkBlock(None, "المادة الأولى", "article_label", (0, 0, 100, 20), 1),),
        hypothesis=(BenchmarkBlock(None, "المادة الأولى", "article_label", (2, 0, 102, 20), 1),),
    )

    assert metrics.cer == 0
    assert metrics.exact_article_number_accuracy == 1


def test_same_page_unrelated_regions_are_not_compared_or_concatenated() -> None:
    gold_box = (0, 0, 100, 20)
    blocks = (
        BenchmarkBlock("target", "المادة الأولى", "article_label", (5, 2, 95, 18), 1),
        BenchmarkBlock("unrelated", "نص لا ينتمي", "paragraph", (0, 200, 100, 240), 2),
    )
    assert assign_blocks_to_gold_region(gold_box, blocks) == (0,)


def test_spatial_assignment_requires_overlap_or_center_inside() -> None:
    blocks = (
        BenchmarkBlock("edge", "edge", "paragraph", (101, 0, 150, 20), 1),
        BenchmarkBlock("center", "center", "paragraph", (20, 2, 80, 18), 2),
    )
    assert assign_blocks_to_gold_region((0, 0, 100, 20), blocks) == (1,)


def test_prediction_bottom_left_geometry_is_canonicalized_before_matching() -> None:
    assert canonicalize_prediction_box(
        (10, 20, 110, 40), origin="bottom-left", page_height=800
    ) == (
        10.0,
        760.0,
        110.0,
        780.0,
    )


def test_anchored_metrics_compare_only_spatially_assigned_blocks() -> None:
    metrics = calculate_anchored_metrics(
        gold_records=(
            {
                "region_id": "r1",
                "gold_text": "المادة الأولى",
                "region_type": "article_label",
                "semantic_article_number": 1,
                "bounding_box": [0, 0, 100, 20],
                "page_number": 1,
            },
            {
                "region_id": "r2",
                "gold_text": "نص مستقل",
                "region_type": "paragraph",
                "semantic_article_number": None,
                "bounding_box": [0, 40, 100, 60],
                "page_number": 1,
            },
        ),
        predicted_blocks=(
            BenchmarkBlock(None, "المادة الأولى", "article_label", (0, 0, 100, 20), 1),
            BenchmarkBlock(None, "غير مرتبط", "paragraph", (0, 200, 100, 220), 2),
            BenchmarkBlock(None, "نص مستقل", "paragraph", (0, 40, 100, 60), 3),
        ),
    )
    assert metrics.cer == 0
    assert metrics.wer == 0
    assert metrics.exact_article_number_accuracy == 1
    assert metrics.semantic_article_number_accuracy == 1
    assert metrics.critical_article_number_errors == 0


def test_anchored_reading_order_uses_predicted_block_order() -> None:
    metrics = calculate_anchored_metrics(
        gold_records=(
            {
                "region_id": "r1",
                "gold_text": "أ",
                "region_type": "paragraph",
                "semantic_article_number": None,
                "bounding_box": [0, 0, 100, 20],
                "page_number": 1,
            },
            {
                "region_id": "r2",
                "gold_text": "ب",
                "region_type": "paragraph",
                "semantic_article_number": None,
                "bounding_box": [0, 40, 100, 60],
                "page_number": 1,
            },
        ),
        predicted_blocks=(
            BenchmarkBlock(None, "ب", "paragraph", (0, 40, 100, 60), 1),
            BenchmarkBlock(None, "أ", "paragraph", (0, 0, 100, 20), 2),
        ),
    )
    assert metrics.reading_order_accuracy == 0
