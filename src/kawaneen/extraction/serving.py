"""Serving-safe Phase 11 extraction adapters for request-scoped text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kawaneen.corpus.models import SourceProvenance
from kawaneen.extraction.contracts import ExtractionResult
from kawaneen.extraction.deterministic import run_deterministic
from kawaneen.extraction.hybrid import assemble_hybrid_result
from kawaneen.extraction.provider import ExtractionProvider
from kawaneen.observability.tracing import NoOpObserver, TraceObserver


@dataclass(frozen=True, slots=True)
class ServingExtractionResponse:
    result: ExtractionResult
    capability_status: Literal["operational_candidates", "experimental_limited"]
    warnings: tuple[str, ...] = ()


class ModelUnavailableError(RuntimeError):
    """The configured hybrid provider cannot serve this request."""


class ServingExtractor:
    def __init__(
        self,
        *,
        provider: ExtractionProvider | None = None,
        observer: TraceObserver | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.provider = provider
        self.observer = observer or NoOpObserver()
        self.provider_name = provider_name
        self.model = model
        self.revision = revision

    def extract(
        self, text: str, *, mode: Literal["deterministic", "hybrid"] = "hybrid"
    ) -> ServingExtractionResponse:
        with self.observer.span(
            "extraction",
            "CHAIN",
            {
                "mode": mode,
                **(
                    {
                        "provider": self.provider_name,
                        "model": self.model,
                        "revision": self.revision,
                    }
                    if mode == "hybrid" and self.provider is not None
                    else {}
                ),
            },
        ) as span:
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
                response = ServingExtractionResponse(base, "operational_candidates")
                span.set_outputs({"capability_status": response.capability_status})
                return response
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
            response = ServingExtractionResponse(
                result,
                "experimental_limited",
                ("PHASE11_HYBRID_EXPERIMENTAL_LIMITED",),
            )
            span.set_outputs({"capability_status": response.capability_status})
            return response
