"""Synthetic extraction response factory kept separate from the HTTP client."""

from __future__ import annotations

import hashlib

from kawaneen.api.contracts import ExtractionResponse
from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.contracts import (
    CandidateRegistry,
    ExtractionResult,
    NormativeRule,
    ExactSourceSpan,
    Modality,
)


def extraction_response(text: str, mode: str) -> ExtractionResponse:
    phrase = "thirty days" if "thirty" in text.lower() else "ثلاثين يوماً"
    start = text.lower().find(phrase.lower()) if phrase.lower() in text.lower() else text.find(phrase)
    if start < 0:
        start = 0
        phrase = text[: min(12, len(text))] or "synthetic"
    span = ExactSourceSpan(
        text=phrase,
        start_char=start,
        end_char=start + len(phrase),
        canonical_unit_id="api-request",
        document_id="api-request",
    )
    rule = NormativeRule(modality=Modality.OBLIGATION, action_span=span, deadline_refs=())
    result = ExtractionResult(
        schema_version="phase11-extraction-v1",
        extractor_version="synthetic-demo",
        configuration="hybrid-qwen-v1" if mode == "hybrid" else "deterministic-v1",
        jurisdiction="SA",
        source_provenance=SourceProvenance(
            source_id="synthetic-demo",
            source_version="phase13",
            source_path="synthetic://phase13",
            source_row=1,
            source_field="text",
        ),
        source_fingerprint=hashlib.sha256(text.encode()).hexdigest(),
        candidate_registry=CandidateRegistry(
            canonical_text=text,
            canonical_unit_id="api-request",
            document_id="api-request",
            candidates=(),
        ),
        obligations=(rule,),
        rules=(rule,),
    )
    return ExtractionResponse(
        request_id="demo-extract",
        result=result,
        capability_status="experimental_limited" if mode == "hybrid" else "operational_candidates",
        latency_ms=0,
        warnings=(),
    )
