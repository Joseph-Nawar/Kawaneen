"""Serving-safe Phase 11 extraction adapters for request-scoped text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.contracts import ExtractionResult
from kawaneen.extraction.deterministic import run_deterministic
from kawaneen.extraction.hybrid import assemble_hybrid_result
from kawaneen.extraction.provider import ExtractionProvider


@dataclass(frozen=True, slots=True)
class ServingExtractionResponse:
    result: ExtractionResult
    capability_status: Literal["operational_candidates", "experimental_limited"]
    warnings: tuple[str, ...] = ()


class ModelUnavailableError(RuntimeError):
    """The configured hybrid provider cannot serve this request."""


class ServingExtractor:
    def __init__(self, *, provider: ExtractionProvider | None = None) -> None:
        self.provider = provider

    def extract(
        self, text: str, *, mode: Literal["deterministic", "hybrid"] = "hybrid"
    ) -> ServingExtractionResponse:
        base = run_deterministic(
            text,
            canonical_unit_id="api-request-unit",
            document_id="api-request-document",
            source_provenance=SourceProvenance(
                source_id="api-request",
                source_version="v1",
                source_path="request-body",
                source_row=1,
                source_field="text",
                split="api",
            ),
        )
        if mode == "deterministic":
            return ServingExtractionResponse(base, "operational_candidates")
        if mode != "hybrid":
            raise ValueError("unsupported extraction mode")
        if self.provider is None:
            raise ModelUnavailableError("hybrid extraction model is unavailable")
        if base.candidate_registry is None:
            raise RuntimeError("deterministic extraction registry is unavailable")
        try:
            raw_proposal = self.provider.propose(text, base.candidate_registry)
        except (ConnectionError, OSError, TimeoutError) as error:
            raise ModelUnavailableError("hybrid extraction model is unavailable") from error
        result = assemble_hybrid_result(text, base, raw_proposal)
        return ServingExtractionResponse(
            result,
            "experimental_limited",
            ("PHASE11_HYBRID_EXPERIMENTAL_LIMITED",),
        )
