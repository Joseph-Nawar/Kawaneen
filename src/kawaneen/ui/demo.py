"""Small synthetic fixtures implementing the public UI client protocol."""

from __future__ import annotations

from typing import cast

from kawaneen.api.contracts import (
    AnswerResponse,
    DocumentDetail,
    DocumentPage,
    ExtractionResponse,
    HealthResponse,
    ModelsResponse,
    SearchResponse,
)
from kawaneen.ui.client import UiClient


class DemoClient(UiClient):
    def search(self, query: str, limit: int = 8) -> SearchResponse:
        english = any(word in query.lower() for word in ("appeal", "deadline", "employment"))
        if english:
            payload = {
                "request_id": "demo-search-en",
                "jurisdiction": "KAWANEEN_DEMO",
                "results": [
                    {
                        "chunk_id": "demo-en-1",
                        "rank": 1,
                        "text": "An appeal may be filed within thirty days from notification.",
                        "document_id": "demo-employment",
                        "document_title": "Employment Procedures Regulation",
                        "article": "Article 42",
                        "page": "18",
                        "source_url": "https://example.invalid/synthetic/employment",
                        "score": 2.4,
                        "score_type": "reranker_raw_logit",
                        "provenance": "both",
                    }
                ],
            }
        else:
            payload = {
                "request_id": "demo-search-ar",
                "jurisdiction": "KAWANEEN_DEMO",
                "results": [
                    {
                        "chunk_id": "demo-ar-1",
                        "rank": 1,
                        "text": "تبدأ مدة الاعتراض من تاريخ التبليغ ولمدة ثلاثين يوماً.",
                        "document_id": "demo-procedure",
                        "document_title": "لائحة الإجراءات التجريبية",
                        "article": "المادة 12",
                        "page": "7",
                        "source_url": None,
                        "score": 2.7,
                        "score_type": "reranker_raw_logit",
                        "provenance": "both",
                    },
                    {
                        "chunk_id": "demo-ar-2",
                        "rank": 2,
                        "text": "يجوز تمديد المدة في الحالات المحددة نظاماً.",
                        "document_id": "demo-procedure",
                        "document_title": "لائحة الإجراءات التجريبية",
                        "article": "المادة 13",
                        "page": "8",
                        "source_url": None,
                        "score": 1.6,
                        "score_type": "reranker_raw_logit",
                        "provenance": "sparse-only",
                    },
                ],
            }
        results = cast(list[dict[str, object]], payload["results"])[:limit]
        return SearchResponse.model_validate(
            {
                **payload,
                "results": results,
                "retrieval": {
                    "strategy": "hybrid_reranked",
                    "sparse_top_k": 50,
                    "dense_top_k": 50,
                    "fused_candidate_count": 20,
                    "reranker_depth": 8,
                    "top_score": cast(float, results[0]["score"]),
                    "hit_count": len(results),
                    "returned_count": len(results),
                    "score_type": "reranker_raw_logit",
                },
                "latency_ms": 0,
                "warnings": (),
            }
        )

    def answer(self, query: str) -> AnswerResponse:
        abstain = any(word in query.lower() for word in ("win", "سأربح", "guarantee"))
        if abstain:
            return AnswerResponse.model_validate(
                {
                    "request_id": "demo-answer-abstention",
                    "jurisdiction": "KAWANEEN_DEMO",
                    "answerable": False,
                    "answer": None,
                    "abstention_reason": "The available evidence cannot establish a case outcome.",
                    "citations": [],
                    "retrieval": self.search(query).retrieval.model_dump(),
                    "latency_ms": 0,
                    "warnings": ["Abstention is an intentional safety decision."],
                }
            )
        search = self.search(query)
        evidence = search.results[0]
        return AnswerResponse.model_validate(
            {
                "request_id": "demo-answer-grounded",
                "jurisdiction": "KAWANEEN_DEMO",
                "answerable": True,
                "answer": evidence.text,
                "abstention_reason": None,
                "citations": [
                    {
                        "evidence_id": "E001",
                        "document_id": evidence.document_id,
                        "document_title": evidence.document_title,
                        "article": evidence.article,
                        "page": evidence.page,
                        "source_url": evidence.source_url,
                        "quoted_text": evidence.text,
                    }
                ],
                "retrieval": search.retrieval.model_dump(),
                "latency_ms": 0,
                "warnings": [],
            }
        )

    def extract(self, text: str, mode: str = "deterministic") -> ExtractionResponse:
        from kawaneen.ui.demo_fixtures import extraction_response

        return extraction_response(text, mode)

    def list_documents(self, offset: int = 0, limit: int = 20) -> DocumentPage:
        items = [
            {
                "document_id": "demo-employment",
                "title": "Employment Procedures Regulation",
                "source_id": "synthetic-demo",
                "jurisdiction": "KAWANEEN_DEMO",
                "unit_count": 2,
            },
            {
                "document_id": "demo-procedure",
                "title": "لائحة الإجراءات التجريبية",
                "source_id": "synthetic-demo",
                "jurisdiction": "KAWANEEN_DEMO",
                "unit_count": 2,
            },
        ]
        page = items[offset : offset + limit]
        return DocumentPage.model_validate(
            {
                "request_id": "demo-documents",
                "items": page,
                "offset": offset,
                "limit": limit,
                "total": len(items),
            }
        )

    def get_document(self, document_id: str) -> DocumentDetail:
        title = (
            "Employment Procedures Regulation"
            if document_id == "demo-employment"
            else "لائحة الإجراءات التجريبية"
        )
        return DocumentDetail.model_validate(
            {
                "request_id": "demo-document-detail",
                "document": {
                    "document_id": document_id,
                    "title": title,
                    "source_id": "synthetic-demo",
                    "jurisdiction": "KAWANEEN_DEMO",
                    "unit_count": 2,
                },
                "units": [
                    {
                        "unit_id": f"{document_id}-u1",
                        "ordinal": 1,
                        "unit_type": "article",
                        "text": "Synthetic heading",
                        "heading_path": (),
                    },
                    {
                        "unit_id": f"{document_id}-u2",
                        "ordinal": 2,
                        "unit_type": "article",
                        "text": "An appeal may be filed within thirty days from notification.",
                        "heading_path": ("Synthetic",),
                    },
                ],
            }
        )

    def health(self) -> HealthResponse:
        return HealthResponse.model_validate(
            {
                "request_id": "demo-health",
                "status": "ready",
                "components": [
                    {
                        "name": "demo-fixtures",
                        "ready": True,
                        "required": True,
                        "detail": "Synthetic data",
                    }
                ],
            }
        )

    def models(self) -> ModelsResponse:
        return ModelsResponse.model_validate(
            {
                "request_id": "demo-models",
                "capabilities": [
                    {
                        "capability": "retrieval",
                        "provider": "synthetic-demo",
                        "model": "fixture",
                        "revision": "tracked",
                        "loaded": True,
                        "ready": True,
                    }
                ],
            }
        )
