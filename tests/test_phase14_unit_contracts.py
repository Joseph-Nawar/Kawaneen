from __future__ import annotations

# ruff: noqa: RUF001
import pytest
from pydantic import ValidationError

from kawaneen.chunking.models import SourceSpan
from kawaneen.corpus.statutory import parse_article_label
from kawaneen.grounding.contracts import (
    CanonicalSourceSpan,
    CitationRequest,
    EvidenceReference,
    SourceRecord,
    VerifiedCitation,
)
from kawaneen.normalization import get_policy
from kawaneen.normalization.policies import normalize_text
from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
from kawaneen.retrieval.hybrid.fusion import fuse_ranked_hits


@pytest.mark.parametrize(
    ("policy_id", "value", "expected"),
    [
        ("arabic-raw-v1", "  أ ـ ب\t١  ", "أ ـ ب ١"),
        ("arabic-light-v1", "  أ ـ ب\t١  ", "ا ب ١"),
        ("arabic-aggressive-v1", "أَ ـ ى ١،", "ا ي 1,"),
    ],
)
def test_normalization_policy_behavior_and_idempotence(
    policy_id: str, value: str, expected: str
) -> None:
    policy = get_policy(policy_id)

    normalized = normalize_text(value, policy)

    assert normalized == expected
    assert normalize_text(normalized, policy) == normalized


@pytest.mark.parametrize(
    ("label", "ordinal", "part"),
    [
        ("المادة ١٢", 12, None),
        ("الماده 12", 12, None),
        ("Article 12", 12, None),
        ("المادة 12 (جزء 2)", 12, 2),
        ("المادة 101", 101, None),
    ],
)
def test_article_reference_variants_extract_only_complete_labels(
    label: str, ordinal: int, part: int | None
) -> None:
    parsed = parse_article_label(label)

    assert parsed.ordinal == ordinal
    assert parsed.part == part


@pytest.mark.parametrize(
    "value",
    ["مادة 12", "فقرة 12", "المادة", "المادة (جزء 2)", "قرار 12"],
)
def test_malformed_or_non_article_text_has_no_false_article_reference(value: str) -> None:
    assert parse_article_label(value).ordinal is None


def test_source_spans_are_non_negative_and_chunk_ids_can_use_valid_boundaries() -> None:
    assert SourceSpan("unit-1", 0, 4).length == 4

    with pytest.raises(ValueError, match="bounds"):
        SourceSpan("unit-1", 5, 4)


def test_weighted_rrf_is_frozen_and_supports_sparse_and_dense_only_candidates() -> None:
    config = FusionConfig(sparse_weight=1.0, dense_weight=0.25, rrf_k=60)
    result = fuse_ranked_hits(
        sparse=(SourceHit("shared", 1.0), SourceHit("sparse", 0.5)),
        dense=(SourceHit("shared", 1.0), SourceHit("dense", 0.5)),
        config=config,
    )

    assert result[0].chunk_id == "shared"
    assert result[0].fused_score == pytest.approx(1 / 61 + 0.25 / 61)
    assert result[1].provenance == "sparse-only"
    assert result[2].provenance == "dense-only"
    assert [
        item.chunk_id
        for item in fuse_ranked_hits(
            sparse=(SourceHit("b", 1.0), SourceHit("a", 1.0)),
            dense=(),
            config=config,
        )
    ] == ["b", "a"]


def test_citation_contract_preserves_exact_provenance_and_forbids_extra_metadata() -> None:
    citation = VerifiedCitation(
        evidence_id="E001",
        document_id="phase14-synthetic-appeals-regulation",
        document_title="Synthetic Appeals Regulation",
        jurisdiction="SA",
        article="المادة ١٢",
        page="1",
        chunk_id="legal-12",
        source_url=None,
        quoted_text="An objection may be submitted within thirty days from notification.",
    )

    assert citation.model_dump(include={"document_id", "article", "page", "quoted_text"}) == {
        "document_id": "phase14-synthetic-appeals-regulation",
        "article": "المادة ١٢",
        "page": "1",
        "quoted_text": "An objection may be submitted within thirty days from notification.",
    }
    with pytest.raises(ValidationError):
        SourceRecord(document_id="doc", unexpected="invented")


@pytest.mark.parametrize(
    "model",
    [
        lambda: CitationRequest(evidence_id="bad", quoted_text="quote"),
        lambda: CanonicalSourceSpan(unit_id="u1", start=4, end=4),
        lambda: EvidenceReference(
            evidence_id="E1",
            unit_id="u1",
            block_id="B001",
            document_id="doc",
            display_text="text",
            source=SourceRecord(document_id="doc"),
            contributing_chunk_ids=("c1",),
            contributing_ranks=(1,),
        ),
    ],
)
def test_invalid_evidence_and_span_references_fail_closed(model) -> None:
    with pytest.raises(ValidationError):
        model()
