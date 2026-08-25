from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from fastapi import FastAPI

from kawaneen.api.app import create_app
from kawaneen.api.runtime import ComponentReadiness, ServiceContainer
from kawaneen.chunking.corpus import Phase5Corpus
from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.chunking.strategies import build_chunks
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.corpus.serving import InMemoryCorpusRepository, ServingDocument, ServingUnit
from kawaneen.corpus.statutory import parse_article_label
from kawaneen.generation.policy import (
    JurisdictionScope,
    PolicyContext,
    SourceStatus,
    evaluate_pre_generation_policy,
)
from kawaneen.generation.serving import ServingAnswerer
from kawaneen.grounding.assembly import ContextAssembler
from kawaneen.grounding.contracts import (
    CanonicalEvidenceUnit,
    CanonicalSourceSpan,
    CitationRequest,
    ClaimDraft,
    ContextPack,
    GeneratedDraft,
    ResolvedChunk,
    RetrievalInput,
    SourceRecord,
)
from kawaneen.grounding.verification import verify_draft
from kawaneen.normalization import get_policy
from kawaneen.retrieval.bm25 import BM25Index
from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.serving import HybridServingRetriever
from kawaneen.retrieval.vector_index import NumpyExactIndex

FIXTURE = Path(__file__).parents[1] / "fixtures" / "phase14" / "synthetic_appeals_regulation.pdf"
DOCUMENT_ID = "phase14-synthetic-appeals-regulation"


def _pdf_lines() -> list[str]:
    from pypdf import PdfReader

    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(FIXTURE)).pages)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _units() -> tuple[CanonicalUnit, ...]:
    lines = _pdf_lines()
    starts = [index for index, line in enumerate(lines) if parse_article_label(line).ordinal]
    result: list[CanonicalUnit] = []
    for row, start in enumerate(starts, start=1):
        end = starts[row] if row < len(starts) else len(lines)
        label = parse_article_label(lines[start])
        assert label.ordinal is not None
        result.append(
            CanonicalUnit(
                unit_id=f"{DOCUMENT_ID}:article-{label.ordinal}",
                document_id=DOCUMENT_ID,
                unit_type=UnitType.ARTICLE,
                text="\n".join(lines[start:end]),
                provenance=SourceProvenance(
                    source_id="phase14-synthetic",
                    source_version="fixture-v1",
                    source_path="tests/fixtures/phase14/synthetic_appeals_regulation.pdf",
                    source_row=row,
                    source_field="article",
                ),
                ordinal=label.ordinal,
            )
        )
    return tuple(result)


class _CorpusResolver:
    chunk_policy_hash = "phase14-synthetic-chunk-policy"

    def __init__(self, units: Sequence[CanonicalUnit], chunks: Sequence[object]) -> None:
        source_units = {
            unit.unit_id: CanonicalEvidenceUnit(
                unit_id=unit.unit_id,
                document_id=unit.document_id,
                ordinal=unit.ordinal,
                display_text=unit.text,
                source=SourceRecord(
                    document_id=unit.document_id,
                    source_id="phase14-synthetic",
                    document_title="Synthetic Appeals Regulation",
                    jurisdiction="SA",
                    article=unit.text.splitlines()[0],
                    page="1",
                ),
            )
            for unit in units
        }
        self._units = source_units
        self._chunks = {
            chunk.chunk_id: (
                tuple(chunk.source_unit_ids),
                tuple(
                    CanonicalSourceSpan(unit_id=span.unit_id, start=span.start, end=span.end)
                    for span in chunk.source_spans
                ),
            )
            for chunk in chunks
        }

    @property
    def units_by_id(self) -> Mapping[str, CanonicalEvidenceUnit]:
        return self._units

    def resolve_chunk(self, chunk_id: str) -> ResolvedChunk:
        value = self._chunks.get(chunk_id)
        if value is None:
            raise ValueError(f"unknown chunk: {chunk_id}")
        unit_ids, spans = value
        return ResolvedChunk(
            chunk_id=chunk_id,
            document_id=DOCUMENT_ID,
            source_unit_ids=unit_ids,
            source_spans=spans,
            units=tuple(self._units[unit_id] for unit_id in unit_ids),
        )


class _Counter:
    identity = "phase14-codepoint-v1"

    def count(self, text: str) -> int:
        return len(text)


def _build_container() -> ServiceContainer:
    units = _units()
    corpus = Phase5Corpus(
        units=units,
        document_ids=frozenset({DOCUMENT_ID}),
        document_count_by_source={"phase14-synthetic": 1},
        source_versions={"phase14-synthetic": "fixture-v1"},
        document_ids_hash="synthetic-document-hash",
        scope_hash="synthetic-scope-hash",
    )
    legal_chunks = build_chunks(
        units, corpus, get_chunk_policy("legal-structure-v1"), get_policy("arabic-light-v1")
    )
    chunks = tuple(
        RetrievalChunk(
            chunk_id=chunk.chunk_id,
            document_id=DOCUMENT_ID,
            source_id="phase14-synthetic",
            unit_type="article",
            display_text=chunk.display_text,
            search_text=chunk.search_text,
            source_unit_ids=chunk.source_unit_ids,
            chunk_policy_hash=chunk.chunk_policy_hash,
            normalization_policy_id=chunk.normalization_policy_id,
            normalization_policy_hash=chunk.normalization_policy_hash,
            token_count=chunk.token_count,
            source_spans=tuple((span.start, span.end) for span in chunk.source_spans),
        )
        for chunk in legal_chunks
    )
    bm25 = BM25Index.build(chunks, "arabic-light-v1")
    vectors = __import__("numpy").eye(len(chunks), dtype="float32")
    vector_index = NumpyExactIndex.build(vectors, [chunk.chunk_id for chunk in chunks])
    by_article = {unit.ordinal: index for index, unit in enumerate(units)}
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def sparse(query: str, top_k: int) -> tuple[SourceHit, ...]:
        return tuple(
            SourceHit(hit.chunk_id, hit.score)
            for hit in bm25.search(query, top_k=top_k)
            if hit.score > 0
        )

    def dense(query: str, top_k: int) -> tuple[SourceHit, ...]:
        lowered = query.casefold()
        if not any(term in lowered for term in ("اعتراض", "objection", "receipt", "استلام")):
            return ()
        ordinal = 12 if "اعتراض" in query or "objection" in lowered else 13
        hits = vector_index.search(vectors[by_article[ordinal]], top_k=top_k)
        return tuple(SourceHit(hit.chunk_id, hit.score) for hit in hits if hit.score > 0)

    def reranker(query: str, candidates: Sequence[object]) -> dict[str, float]:
        preferred = "مهلة" in query or "deadline" in query.casefold()
        return {
            candidate.chunk_id: (
                100.0
                if preferred and "المادة ١٢" in chunk_by_id[candidate.chunk_id].display_text
                else float(90 - index)
            )
            for index, candidate in enumerate(candidates)
        }

    retriever = HybridServingRetriever(
        chunks={chunk.chunk_id: chunk for chunk in chunks},
        sparse_search=sparse,
        dense_search=dense,
        reranker=reranker,
        fusion_config=FusionConfig(sparse_weight=1.0, dense_weight=0.25),
        metadata={
            chunk.chunk_id: {
                "document_title": "Synthetic Appeals Regulation",
                "article": next(
                    unit.text.splitlines()[0]
                    for unit in units
                    if unit.unit_id in chunk.source_unit_ids
                ),
                "page": "1",
            }
            for chunk in chunks
        },
    )
    resolver = _CorpusResolver(units, legal_chunks)
    assembler = ContextAssembler(resolver, _Counter(), max_context_tokens=10_000)

    def build_context(query: str, retrieval: object) -> ContextPack:
        evidence = retrieval.evidence  # type: ignore[attr-defined]
        ranked = tuple(
            RetrievalInput(query_id="phase14-e2e", rank=index, chunk_id=item.chunk_id)
            for index, item in enumerate(evidence, start=1)
        )
        return assembler.assemble(
            query_id="phase14-e2e",
            ranked_inputs=ranked,
            phase8_selection_sha256="a" * 64,
            canonical_corpus_hash="b" * 64,
        )

    def policy(query: str, context: object):
        return evaluate_pre_generation_policy(
            query,
            PolicyContext(
                context_pack=context,
                scope=JurisdictionScope(
                    active_jurisdiction="SA",
                    authoritative_jurisdiction="SA",
                    allowed_jurisdictions=("SA",),
                    mode="single",
                    required=True,
                ),
                source_status=SourceStatus.ACTIVE,
                source_status_available=True,
            ),
        )

    def generator(query: str, context: ContextPack) -> GeneratedDraft | None:
        if not context.evidence:
            return None
        evidence = context.evidence[0]
        return GeneratedDraft(
            answer_text=evidence.display_text,
            claims=(
                ClaimDraft(
                    claim_id="C001",
                    claim_text=evidence.display_text,
                    citations=(
                        CitationRequest(
                            evidence_id=evidence.evidence_id,
                            quoted_text=evidence.display_text,
                        ),
                    ),
                ),
            ),
        )

    answerer = ServingAnswerer(
        retriever=lambda query, limit=8: retriever.search(query, limit),
        context_builder=build_context,
        policy_evaluator=policy,
        generator=generator,
        verifier=lambda context, draft: verify_draft(context, draft, resolver),
    )
    documents = InMemoryCorpusRepository(
        (
            ServingDocument(
                document_id=DOCUMENT_ID,
                title="Synthetic Appeals Regulation",
                source_id="phase14-synthetic",
                units=tuple(
                    ServingUnit(
                        unit_id=unit.unit_id,
                        unit_type=unit.unit_type.value,
                        text=unit.text,
                        ordinal=unit.ordinal,
                    )
                    for unit in units
                ),
            ),
        )
    )
    components = (
        ComponentReadiness("corpus", True, True),
        ComponentReadiness("retrieval", True, True),
        ComponentReadiness("answer", True, True),
        ComponentReadiness("extraction_deterministic", True, False),
        ComponentReadiness("extraction_hybrid", False, False, "not used by public E2E"),
    )
    return ServiceContainer(
        retriever=retriever,
        answerer=answerer,
        corpus=documents,
        components=components,
        initializer=lambda: None,
    )


app: FastAPI = create_app(lambda: _build_container())
