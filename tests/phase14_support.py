from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kawaneen.api.runtime import ComponentReadiness, ServiceContainer
from kawaneen.chunking.corpus import Phase5Corpus
from kawaneen.chunking.models import LegalChunk
from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.chunking.strategies import build_chunks
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.corpus.serving import InMemoryCorpusRepository, ServingDocument, ServingUnit
from kawaneen.corpus.statutory import parse_article_label
from kawaneen.generation.answerability import SourceEligibility, evaluate_stage_d_policy
from kawaneen.generation.contracts import AbstentionReason
from kawaneen.generation.policy import (
    JurisdictionScope,
    PolicyContext,
    SourceStatus,
)
from kawaneen.generation.serving import ServingAnswerer, ServingAnswerResult
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
from kawaneen.parsing.health import probe_pdf
from kawaneen.retrieval.bm25 import BM25Index
from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.serving import HybridServingRetriever, ServingRetrievalResult
from kawaneen.retrieval.tokenization import represent
from kawaneen.retrieval.vector_index import NumpyExactIndex

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "phase14" / "synthetic_appeals_regulation.pdf"
CORRUPT_FIXTURE = ROOT / "tests" / "fixtures" / "phase14" / "corrupt.pdf"
DOCUMENT_ID = "phase14-synthetic-appeals-regulation"
SOURCE_ID = "phase14-synthetic"
NORMALIZATION_POLICY_ID = "arabic-light-v1"
CHUNK_POLICY_ID = "legal-structure-v1"
VECTOR_DIMENSION = 128


def extract_synthetic_pdf_text(path: Path = FIXTURE) -> str:
    health = probe_pdf(path)
    if not health or any(page.text_chars == 0 for page in health):
        raise ValueError("synthetic fixture must contain embedded text on every page")
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def build_synthetic_units(path: Path = FIXTURE) -> tuple[CanonicalUnit, ...]:
    lines = [line.strip() for line in extract_synthetic_pdf_text(path).splitlines() if line.strip()]
    starts = [index for index, line in enumerate(lines) if parse_article_label(line).ordinal]
    units: list[CanonicalUnit] = []
    for row, start in enumerate(starts, start=1):
        end = starts[row] if row < len(starts) else len(lines)
        label = parse_article_label(lines[start])
        assert label.ordinal is not None
        units.append(
            CanonicalUnit(
                unit_id=f"{DOCUMENT_ID}:article-{label.ordinal}",
                document_id=DOCUMENT_ID,
                unit_type=UnitType.ARTICLE,
                text="\n".join(lines[start:end]),
                provenance=SourceProvenance(
                    source_id=SOURCE_ID,
                    source_version="fixture-v1",
                    source_path="tests/fixtures/phase14/synthetic_appeals_regulation.pdf",
                    source_row=row,
                    source_field="article",
                ),
                ordinal=label.ordinal,
            )
        )
    if not units:
        raise ValueError("synthetic fixture has no article units")
    return tuple(units)


def build_synthetic_corpus(units: tuple[CanonicalUnit, ...]) -> Phase5Corpus:
    return Phase5Corpus(
        units=units,
        document_ids=frozenset({DOCUMENT_ID}),
        document_count_by_source={SOURCE_ID: 1},
        source_versions={SOURCE_ID: "fixture-v1"},
        document_ids_hash=hashlib.sha256(DOCUMENT_ID.encode()).hexdigest(),
        scope_hash=hashlib.sha256(",".join(unit.unit_id for unit in units).encode()).hexdigest(),
    )


def build_synthetic_legal_chunks(
    units: tuple[CanonicalUnit, ...],
) -> tuple[LegalChunk, ...]:
    return build_chunks(
        units,
        build_synthetic_corpus(units),
        get_chunk_policy(CHUNK_POLICY_ID),
        get_policy(NORMALIZATION_POLICY_ID),
    )


def to_retrieval_chunks(
    legal_chunks: Sequence[LegalChunk],
) -> tuple[RetrievalChunk, ...]:
    return tuple(
        RetrievalChunk(
            chunk_id=chunk.chunk_id,
            document_id=DOCUMENT_ID,
            source_id=SOURCE_ID,
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


class SyntheticCorpusResolver:
    chunk_policy_hash = get_chunk_policy(CHUNK_POLICY_ID).policy_hash

    def __init__(self, units: Sequence[CanonicalUnit], chunks: Sequence[LegalChunk]) -> None:
        self._units = {
            unit.unit_id: CanonicalEvidenceUnit(
                unit_id=unit.unit_id,
                document_id=unit.document_id,
                ordinal=unit.ordinal,
                display_text=unit.text,
                source=SourceRecord(
                    document_id=unit.document_id,
                    source_id=SOURCE_ID,
                    document_title="Synthetic Appeals Regulation",
                    jurisdiction="SA",
                    article=unit.text.splitlines()[0],
                    page="1",
                ),
            )
            for unit in units
        }
        self._chunks = {
            chunk.chunk_id: (
                tuple(chunk.source_unit_ids),
                tuple(
                    CanonicalSourceSpan(
                        unit_id=span.unit_id,
                        start=span.start,
                        end=span.end,
                    )
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


class _CodepointCounter:
    identity = "phase14-codepoint-v1"

    def count(self, text: str) -> int:
        return len(text)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(represent(text, NORMALIZATION_POLICY_ID).tokens)


def deterministic_embedding(text: str) -> np.ndarray:
    vector = np.zeros(VECTOR_DIMENSION, dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % VECTOR_DIMENSION
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return (vector / norm).astype(np.float32, copy=False)


@dataclass
class Phase14Stack:
    units: tuple[CanonicalUnit, ...]
    legal_chunks: tuple[LegalChunk, ...]
    chunks: tuple[RetrievalChunk, ...]
    bm25: BM25Index
    vector_index: NumpyExactIndex
    retriever: HybridServingRetriever
    resolver: SyntheticCorpusResolver
    assembler: ContextAssembler
    answerer: ServingAnswerer
    article_by_chunk: Mapping[str, int]
    search_calls: list[tuple[str, int, str]]

    def article_ordinal(self, chunk_id: str) -> int:
        return self.article_by_chunk[chunk_id]

    @property
    def chunks_by_id(self) -> Mapping[str, RetrievalChunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}

    def answer(self, query: str) -> ServingAnswerResult:
        return self.answerer.answer(query)

    def context_for(self, query: str, retrieval: ServingRetrievalResult) -> ContextPack:
        query_id = hashlib.sha256(query.encode("utf-8")).hexdigest()
        ranked = tuple(
            RetrievalInput(query_id=query_id, rank=item.rank, chunk_id=item.chunk_id)
            for item in retrieval.evidence
        )
        return self.assembler.assemble(
            query_id=query_id,
            ranked_inputs=ranked,
            phase8_selection_sha256="a" * 64,
            canonical_corpus_hash="b" * 64,
        )


def build_phase14_stack() -> Phase14Stack:
    units = build_synthetic_units()
    legal_chunks = build_synthetic_legal_chunks(units)
    chunks = to_retrieval_chunks(legal_chunks)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    article_by_chunk = {
        chunk.chunk_id: next(
            unit.ordinal
            for unit in units
            if unit.unit_id in chunk.source_unit_ids and unit.ordinal is not None
        )
        for chunk in chunks
    }
    vectors = np.vstack([deterministic_embedding(chunk.display_text) for chunk in chunks])
    vector_index = NumpyExactIndex.build(vectors, [chunk.chunk_id for chunk in chunks])
    bm25 = BM25Index.build(chunks, NORMALIZATION_POLICY_ID)
    search_calls: list[tuple[str, int, str]] = []

    def sparse(query: str, top_k: int) -> tuple[SourceHit, ...]:
        search_calls.append((query, top_k, "sparse"))
        return tuple(
            SourceHit(hit.chunk_id, hit.score)
            for hit in bm25.search(query, top_k=top_k)
            if hit.score > 0.0
        )

    def dense(query: str, top_k: int) -> tuple[SourceHit, ...]:
        search_calls.append((query, top_k, "dense"))
        vector = deterministic_embedding(query)
        if not np.any(vector):
            return ()
        return tuple(
            SourceHit(hit.chunk_id, hit.score)
            for hit in vector_index.search(vector, top_k=top_k)
            if hit.score > 0.0
        )

    def reranker(query: str, candidates: Sequence[object]) -> dict[str, float]:
        query_tokens = _tokens(query)
        return {
            candidate.chunk_id: float(
                len(query_tokens & _tokens(chunk_by_id[candidate.chunk_id].display_text)) * 100.0
                + (len(candidates) - index)
            )
            for index, candidate in enumerate(candidates)
        }

    metadata = {
        chunk.chunk_id: {
            "document_title": "Synthetic Appeals Regulation",
            "article": next(
                unit.text.splitlines()[0] for unit in units if unit.unit_id in chunk.source_unit_ids
            ),
            "page": "1",
        }
        for chunk in chunks
    }
    retriever = HybridServingRetriever(
        chunks=chunk_by_id,
        sparse_search=sparse,
        dense_search=dense,
        reranker=reranker,
        fusion_config=FusionConfig(sparse_weight=1.0, dense_weight=0.25),
        metadata=metadata,
    )
    resolver = SyntheticCorpusResolver(units, legal_chunks)
    assembler = ContextAssembler(resolver, _CodepointCounter(), max_context_tokens=10_000)

    def policy(query: str, context: object):
        from kawaneen.grounding.contracts import ContextPack

        if not isinstance(context, ContextPack):
            raise ValueError("phase14 context is invalid")
        base = evaluate_stage_d_policy(
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
            source_registry={
                SOURCE_ID: SourceEligibility(
                    source_id=SOURCE_ID,
                    source_type="regulation",
                    source_role="primary legal source",
                    authority_level="official",
                    decision="approved",
                    scope_terms=("SA", "regulation"),
                )
            },
        )
        if not base.allowed:
            return base
        query_tokens = _tokens(query)
        evidence_tokens = frozenset(
            token for evidence in context.evidence for token in _tokens(evidence.display_text)
        )
        overlap = len(query_tokens & evidence_tokens)
        if overlap < 2:
            return base.model_copy(
                update={
                    "allowed": False,
                    "reason": AbstentionReason.NO_CONTEXT,
                    "detail": "synthetic answerability requires two matching normalized terms",
                }
            )
        return base

    def generator(query: str, context: object) -> GeneratedDraft | None:
        if not isinstance(context, ContextPack) or not context.evidence:
            return None
        evidence = min(context.evidence, key=lambda item: min(item.contributing_ranks))
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
        context_builder=lambda query, retrieval: Phase14Stack.context_for(stack, query, retrieval),
        policy_evaluator=policy,
        generator=generator,
        verifier=lambda context, draft: verify_draft(context, draft, resolver),
    )
    stack = Phase14Stack(
        units=units,
        legal_chunks=legal_chunks,
        chunks=chunks,
        bm25=bm25,
        vector_index=vector_index,
        retriever=retriever,
        resolver=resolver,
        assembler=assembler,
        answerer=answerer,
        article_by_chunk=article_by_chunk,
        search_calls=search_calls,
    )
    return stack


def build_phase14_service_container() -> ServiceContainer:
    from kawaneen.api.runtime import ServiceContainer

    stack = build_phase14_stack()
    documents = InMemoryCorpusRepository(
        (
            ServingDocument(
                document_id=DOCUMENT_ID,
                title="Synthetic Appeals Regulation",
                source_id=SOURCE_ID,
                units=tuple(
                    ServingUnit(
                        unit_id=unit.unit_id,
                        unit_type=unit.unit_type.value,
                        text=unit.text,
                        ordinal=unit.ordinal,
                    )
                    for unit in stack.units
                ),
            ),
        )
    )
    return ServiceContainer(
        retriever=stack.retriever,
        answerer=stack.answerer,
        corpus=documents,
        components=(
            ComponentReadiness("corpus", True, True),
            ComponentReadiness("retrieval", True, True),
            ComponentReadiness("answer", True, True),
            ComponentReadiness("extraction_deterministic", True, False),
            ComponentReadiness("extraction_hybrid", False, False, "not used by public E2E"),
        ),
        initializer=lambda: None,
    )
