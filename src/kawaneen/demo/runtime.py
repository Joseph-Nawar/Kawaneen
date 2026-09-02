from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np

from kawaneen.api.runtime import ComponentReadiness, ModelSnapshot, ServiceContainer
from kawaneen.core.config import Settings
from kawaneen.demo.corpus import DEFAULT_DEMO_ROOT, load_demo_corpus
from kawaneen.demo.limits import DemoRequestLimiter
from kawaneen.demo.retrieval import RERANKER_MODEL_ID, RERANKER_REVISION, DemoRetriever
from kawaneen.extraction.serving import ServingExtractor
from kawaneen.generation.serving import ServingAnswerResult
from kawaneen.grounding.contracts import VerifiedCitation
from kawaneen.observability.tracing import NoOpObserver


class DemoAnswerer:
    def __init__(self, retriever: DemoRetriever) -> None:
        self.retriever = retriever

    def answer(self, query: str) -> ServingAnswerResult:
        result = self.retriever.search(query, 5)
        if not result.evidence:
            return ServingAnswerResult(False, None, "INSUFFICIENT_DEMO_EVIDENCE", (), result)
        evidence = result.evidence[0]
        citation = VerifiedCitation(
            evidence_id="E001",
            document_id=evidence.document_id,
            document_title=evidence.document_title,
            jurisdiction="KAWANEEN_DEMO",
            article=evidence.article,
            page=evidence.page,
            chunk_id=evidence.chunk_id,
            source_url=None,
            quoted_text=evidence.text,
        )
        return ServingAnswerResult(True, evidence.text, None, (citation,), result)


def create_demo_container(
    *,
    root: Path = DEFAULT_DEMO_ROOT,
    query_encoder: Callable[[str], np.ndarray] | None = None,
    request_rate_limit: int = 30,
    use_reranker: bool = True,
) -> ServiceContainer:
    corpus = load_demo_corpus(root)
    reranker = None
    if use_reranker:
        from kawaneen.retrieval.hybrid.reranker import BGERerankerAdapter

        reranker = BGERerankerAdapter(revision=RERANKER_REVISION, device="cpu", max_length=512)
    retriever = DemoRetriever(corpus, query_encoder=query_encoder, reranker=reranker)
    answerer = DemoAnswerer(retriever)
    components = (
        ComponentReadiness("corpus", True, False, "synthetic KAWANEEN_DEMO corpus"),
        ComponentReadiness("retrieval", True, True),
        ComponentReadiness("answer", True, True, "deterministic evidence-first response"),
        ComponentReadiness("extraction_deterministic", True, False),
        ComponentReadiness("extraction_hybrid", False, False, "public demo has no LLM extraction"),
    )
    models = (
        ModelSnapshot(
            "retrieval-dense",
            "huggingface",
            "intfloat/multilingual-e5-small",
            cast(str, corpus.manifest["model_revision"]),
            True,
            True,
        ),
        ModelSnapshot(
            "retrieval-reranker",
            "huggingface",
            RERANKER_MODEL_ID if use_reranker else None,
            RERANKER_REVISION if use_reranker else None,
            False,
            use_reranker,
        ),
    )
    return ServiceContainer(
        retriever=retriever,
        answerer=answerer,
        extractor=ServingExtractor(observer=NoOpObserver()),
        corpus=None,
        components=components,
        model_metadata=models,
        settings=Settings(),
        observer=NoOpObserver(),
        public_demo=True,
        demo_guard=DemoRequestLimiter(rate_limit=request_rate_limit),
    )


def create_demo_app(
    *,
    root: Path = DEFAULT_DEMO_ROOT,
    query_encoder: Callable[[str], np.ndarray] | None = None,
    request_rate_limit: int = 30,
    use_reranker: bool = True,
):
    from kawaneen.api.app import ApiSettings, create_app

    return create_app(
        lambda: create_demo_container(
            root=root,
            query_encoder=query_encoder,
            request_rate_limit=request_rate_limit,
            use_reranker=use_reranker,
        ),
        settings=Settings(),
        api_settings=ApiSettings(
            search_timeout_seconds=18, answer_timeout_seconds=18, extract_timeout_seconds=18
        ),
    )


__all__ = ["DemoAnswerer", "create_demo_app", "create_demo_container"]
