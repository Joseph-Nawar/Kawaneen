"""Deterministic canonical-span to legal-structure-v1 qrel derivation."""

from __future__ import annotations

from collections import defaultdict

from kawaneen.chunking.models import deterministic_chunk_id
from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.chunking.structure import split_exact_spans
from kawaneen.evaluation.corpus import EvaluationCorpus
from kawaneen.evaluation.models import ChunkQrel, DatasetItem, RelevanceGrade


def map_items_to_chunks(
    items: tuple[DatasetItem, ...], corpus: EvaluationCorpus
) -> tuple[DatasetItem, ...]:
    policy = get_chunk_policy("legal-structure-v1")
    units = {unit.unit_id: unit for unit in corpus.units}
    result: list[DatasetItem] = []
    for item in items:
        if not item.evidence_groups:
            result.append(
                item.model_copy(update={"chunk_policy_hash": policy.policy_hash, "chunk_qrels": ()})
            )
            continue
        grades: defaultdict[str, RelevanceGrade] = defaultdict(lambda: RelevanceGrade.IRRELEVANT)
        for group in item.evidence_groups:
            for evidence in group.spans:
                unit = units.get(evidence.unit_id)
                if unit is None:
                    raise ValueError(
                        f"evidence unit is outside evaluation corpus: {evidence.unit_id}"
                    )
                for chunk_span in split_exact_spans(unit.text, unit.unit_id):
                    if chunk_span.start < evidence.end and evidence.start < chunk_span.end:
                        chunk_id = deterministic_chunk_id(
                            "legal-structure-v1",
                            unit.unit_id,
                            (chunk_span,),
                            policy.policy_hash,
                        )
                        grades[chunk_id] = max(grades[chunk_id], evidence.grade)
        qrels = tuple(
            ChunkQrel(chunk_id=chunk_id, grade=grade)
            for chunk_id, grade in sorted(grades.items())
            if grade > RelevanceGrade.IRRELEVANT
        )
        result.append(
            item.model_copy(
                update={
                    "chunk_policy_hash": policy.policy_hash,
                    "chunk_qrels": qrels,
                }
            )
        )
    return tuple(result)
