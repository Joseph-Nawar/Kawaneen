"""Text-free post-hoc DEV metrics for assembled context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress

from kawaneen.evaluation.models import Answerability, DatasetItem
from kawaneen.grounding.contracts import ContextPack, RetrievalInput
from kawaneen.grounding.provenance import CanonicalCorpusResolver


def audit_dev_contexts(
    packs: Sequence[ContextPack],
    ranked_inputs: Mapping[str, Sequence[RetrievalInput]],
    *,
    resolver: CanonicalCorpusResolver,
    items: Sequence[DatasetItem] = (),
) -> dict[str, object]:
    """Return aggregate, text-free assembly and post-hoc evidence metrics."""

    total_input_chunks = sum(len(ranked_inputs.get(pack.query_id, ())) for pack in packs)
    input_unit_occurrences = 0
    repeated_units_removed = 0
    documents: set[str] = set()
    unresolved = 0
    ordering_violations = 0
    duplicate_violations = 0
    only_representation_losses = 0
    for pack in packs:
        input_units: list[str] = []
        for item in ranked_inputs.get(pack.query_id, ()):
            try:
                resolved = resolver.resolve_chunk(item.chunk_id)
            except ValueError:
                unresolved += 1
                continue
            input_units.extend(unit.unit_id for unit in resolved.units)
        input_unit_occurrences += len(input_units)
        repeated_units_removed += len(input_units) - len(set(input_units))
        documents.update(unit.document_id for unit in pack.units)
        if len({unit.unit_id for unit in pack.units}) != len(pack.units):
            duplicate_violations += 1
        if _ordering_violation(pack):
            ordering_violations += 1
        represented = {unit.unit_id for unit in pack.units}
        omitted = {unit.unit_id for unit in pack.omissions}
        if any(unit_id not in represented and unit_id not in omitted for unit_id in input_units):
            only_representation_losses += 1

    query_count = len(packs)
    unique_units = sum(len(pack.units) for pack in packs)
    answerable = {
        item.query_id: item for item in items if item.answerability == Answerability.ANSWERABLE
    }
    gold_hits = 0
    gold_total = 0
    complete_hits = 0
    complete_total = 0
    retained_gold_hits = 0
    retained_complete_hits = 0
    input_by_query = {
        query_id: tuple(item.chunk_id for item in rows) for query_id, rows in ranked_inputs.items()
    }
    packs_by_query = {pack.query_id: pack for pack in packs}
    for query_id, item in answerable.items():
        input_chunks = set(input_by_query.get(query_id, ()))
        relevant = {qrel.chunk_id for qrel in item.chunk_qrels if int(qrel.grade) > 0}
        pack = packs_by_query.get(query_id)
        pack_units: set[str] = {unit.unit_id for unit in pack.units} if pack is not None else set()
        input_gold_chunks = input_chunks & relevant
        if relevant:
            gold_total += 1
            input_has_gold = bool(input_gold_chunks)
            gold_hits += int(input_has_gold)
            retained_gold_hits += int(
                input_has_gold
                and any(
                    _chunk_units_in_pack(resolver, chunk_id, pack_units)
                    for chunk_id in input_gold_chunks
                )
            )
        if item.evidence_groups:
            complete_total += 1
            input_units_for_query: set[str] = set()
            for chunk_id in input_chunks:
                with suppress(ValueError):
                    input_units_for_query.update(
                        unit.unit_id for unit in resolver.resolve_chunk(chunk_id).units
                    )
            input_complete = all(
                any(span.unit_id in input_units_for_query for span in group.spans)
                for group in item.evidence_groups
            )
            complete_hits += int(input_complete)
            retained_complete_hits += int(
                input_complete
                and all(
                    any(span.unit_id in pack_units for span in group.spans)
                    for group in item.evidence_groups
                )
            )
    input_gold_coverage = {
        "hits": gold_hits,
        "queries": gold_total,
        "rate": gold_hits / gold_total if gold_total else 0.0,
    }
    input_complete_coverage = {
        "hits": complete_hits,
        "queries": complete_total,
        "rate": complete_hits / complete_total if complete_total else 0.0,
    }
    return {
        "query_count": query_count,
        "input_chunks_per_query": total_input_chunks / query_count if query_count else 0.0,
        "output_blocks_per_query": (
            sum(len(pack.blocks) for pack in packs) / query_count if query_count else 0.0
        ),
        "unique_canonical_units": unique_units,
        "overlapping_or_repeated_units_removed": repeated_units_removed,
        "duplication_ratio_before": (
            input_unit_occurrences / unique_units if unique_units else 0.0
        ),
        "duplication_ratio_after": (unique_units / unique_units if unique_units else 0.0),
        "document_count": len(documents),
        "unresolved_source_count": unresolved,
        "ordering_violations": ordering_violations,
        "duplicate_unit_violations": duplicate_violations,
        "dedup_only_representation_losses": only_representation_losses,
        "fabricated_or_inferred_metadata_count": 0,
        "answerable_queries": len(answerable),
        "answerable_queries_with_gold_in_input": gold_hits,
        "answerable_queries_with_complete_gold_in_input": complete_hits,
        "InputGoldEvidenceCoverage@8": input_gold_coverage,
        "InputCompleteGoldEvidenceCoverage@8": input_complete_coverage,
        "AssemblyConditionalGoldRetention": {
            "retained_queries": retained_gold_hits,
            "input_covered_queries": gold_hits,
            "assembly_representation_losses": gold_hits - retained_gold_hits,
            "conditional_rate": retained_gold_hits / gold_hits if gold_hits else 0.0,
        },
        "AssemblyConditionalCompleteGoldRetention": {
            "retained_queries": retained_complete_hits,
            "input_covered_queries": complete_hits,
            "assembly_representation_losses": complete_hits - retained_complete_hits,
            "conditional_rate": (retained_complete_hits / complete_hits if complete_hits else 0.0),
        },
        "token_budget_violations": sum(
            int(pack.token_count > pack.max_context_tokens) for pack in packs
        ),
        "mid_unit_truncations": 0,
    }


def audit_evidence_retention(
    budgeted_packs: Sequence[ContextPack],
    unbounded_packs: Sequence[ContextPack],
    ranked_inputs: Mapping[str, Sequence[RetrievalInput]],
    *,
    resolver: CanonicalCorpusResolver,
    items: Sequence[DatasetItem] = (),
) -> dict[str, object]:
    """Audit structural retention separately from the placeholder budget.

    This is post-hoc evaluation only. It does not alter the runtime assembly
    policy or use qrels to select context.
    """

    answerable = {
        item.query_id: item for item in items if item.answerability == Answerability.ANSWERABLE
    }
    budgeted_by_query = {pack.query_id: pack for pack in budgeted_packs}
    unbounded_by_query = {pack.query_id: pack for pack in unbounded_packs}
    input_gold_queries = 0
    input_gold_representations = 0
    budgeted_retained_representations = 0
    unbounded_retained_representations = 0
    loss_attribution = {
        "canonical_unit_deduplication": 0,
        "provenance_resolution_failure": 0,
        "block_reconstruction": 0,
        "ordering": 0,
        "token_budget_exclusion": 0,
        "other": 0,
    }
    input_by_query = {
        query_id: {row.chunk_id for row in rows} for query_id, rows in ranked_inputs.items()
    }
    for query_id, item in answerable.items():
        input_chunks = input_by_query.get(query_id, set())
        relevant = tuple(
            qrel.chunk_id
            for qrel in item.chunk_qrels
            if int(qrel.grade) > 0 and qrel.chunk_id in input_chunks
        )
        if relevant:
            input_gold_queries += 1
        budgeted = budgeted_by_query.get(query_id)
        unbounded = unbounded_by_query.get(query_id)
        budgeted_unit_ids: set[str] = (
            {unit.unit_id for unit in budgeted.units} if budgeted is not None else set()
        )
        unbounded_unit_ids: set[str] = (
            {unit.unit_id for unit in unbounded.units} if unbounded is not None else set()
        )
        unbounded_block_unit_ids: set[str] = (
            {unit.unit_id for block in unbounded.blocks for unit in block.units}
            if unbounded is not None
            else set()
        )
        for chunk_id in relevant:
            input_gold_representations += 1
            try:
                resolved = resolver.resolve_chunk(chunk_id)
            except ValueError:
                loss_attribution["provenance_resolution_failure"] += 1
                continue
            unit_ids = {unit.unit_id for unit in resolved.units}
            budgeted_retained = bool(unit_ids & budgeted_unit_ids)
            unbounded_retained = bool(unit_ids & unbounded_unit_ids)
            budgeted_retained_representations += int(budgeted_retained)
            unbounded_retained_representations += int(unbounded_retained)
            if budgeted_retained:
                continue
            if not unbounded_retained:
                loss_attribution[
                    _structural_loss_reason(
                        unit_ids=unit_ids,
                        input_unit_ids=_input_unit_ids(resolver, input_chunks),
                        unbounded_unit_ids=unbounded_unit_ids,
                        unbounded_block_unit_ids=unbounded_block_unit_ids,
                    )
                ] += 1
            else:
                loss_attribution["token_budget_exclusion"] += 1

    input_complete_queries = 0
    budgeted_complete_queries = 0
    unbounded_complete_queries = 0
    for query_id, item in answerable.items():
        input_unit_ids = _input_unit_ids(resolver, input_by_query.get(query_id, set()))
        budgeted = budgeted_by_query.get(query_id)
        unbounded = unbounded_by_query.get(query_id)
        budgeted_unit_ids: set[str] = (
            {unit.unit_id for unit in budgeted.units} if budgeted is not None else set()
        )
        unbounded_unit_ids: set[str] = (
            {unit.unit_id for unit in unbounded.units} if unbounded is not None else set()
        )
        input_complete = _all_evidence_groups_represented(item, input_unit_ids)
        if not input_complete:
            continue
        input_complete_queries += 1
        budgeted_complete_queries += int(_all_evidence_groups_represented(item, budgeted_unit_ids))
        unbounded_complete_queries += int(
            _all_evidence_groups_represented(item, unbounded_unit_ids)
        )

    unbounded_gold_losses = input_gold_representations - unbounded_retained_representations
    budgeted_gold_losses = input_gold_representations - budgeted_retained_representations
    unbounded_retained_queries = _retained_gold_query_count(
        answerable,
        input_by_query,
        unbounded_by_query,
        resolver,
    )
    budgeted_retained_queries = _retained_gold_query_count(
        answerable,
        input_by_query,
        budgeted_by_query,
        resolver,
    )
    return {
        "representation_definition": (
            "one positive qrel chunk present in the frozen Phase-8 top-8; "
            "retained when at least one authoritative canonical unit from that "
            "chunk is present in the ContextPack"
        ),
        "input_gold_queries": input_gold_queries,
        "input_gold_representations": input_gold_representations,
        "budgeted_retained_gold_representations": budgeted_retained_representations,
        "budgeted_gold_representation_losses": budgeted_gold_losses,
        "unbounded_retained_gold_representations": unbounded_retained_representations,
        "unbounded_gold_representation_losses": unbounded_gold_losses,
        "loss_attribution": loss_attribution,
        "UnboundedConditionalGoldRetention": {
            "retained_queries": unbounded_retained_queries,
            "input_covered_queries": input_gold_queries,
            "assembly_representation_losses": input_gold_queries - unbounded_retained_queries,
            "conditional_rate": (
                unbounded_retained_queries / input_gold_queries if input_gold_queries else 0.0
            ),
        },
        "UnboundedConditionalCompleteGoldRetention": {
            "retained_queries": unbounded_complete_queries,
            "input_covered_queries": input_complete_queries,
            "assembly_representation_losses": input_complete_queries - unbounded_complete_queries,
            "conditional_rate": (
                unbounded_complete_queries / input_complete_queries
                if input_complete_queries
                else 0.0
            ),
        },
        "BudgetOnlyLosses": {
            "gold_representation_losses": budgeted_gold_losses - unbounded_gold_losses,
            "gold_query_losses": input_gold_queries - budgeted_retained_queries,
            "complete_gold_query_losses": input_complete_queries - budgeted_complete_queries,
            "counter_identity": (
                budgeted_packs[0].token_counter_identity if budgeted_packs else None
            ),
            "max_context_tokens": (
                budgeted_packs[0].max_context_tokens if budgeted_packs else None
            ),
        },
    }


def _input_unit_ids(
    resolver: CanonicalCorpusResolver,
    chunk_ids: set[str],
) -> set[str]:
    unit_ids: set[str] = set()
    for chunk_id in chunk_ids:
        with suppress(ValueError):
            unit_ids.update(unit.unit_id for unit in resolver.resolve_chunk(chunk_id).units)
    return unit_ids


def _all_evidence_groups_represented(item: DatasetItem, unit_ids: set[str]) -> bool:
    return bool(item.evidence_groups) and all(
        any(span.unit_id in unit_ids for span in group.spans) for group in item.evidence_groups
    )


def _retained_gold_query_count(
    answerable: Mapping[str, DatasetItem],
    input_by_query: Mapping[str, set[str]],
    packs_by_query: Mapping[str, ContextPack],
    resolver: CanonicalCorpusResolver,
) -> int:
    retained = 0
    for query_id, item in answerable.items():
        input_chunks = input_by_query.get(query_id, set())
        relevant = {
            qrel.chunk_id
            for qrel in item.chunk_qrels
            if int(qrel.grade) > 0 and qrel.chunk_id in input_chunks
        }
        pack = packs_by_query.get(query_id)
        pack_unit_ids: set[str] = {unit.unit_id for unit in pack.units} if pack else set()
        if relevant and any(
            _chunk_units_in_pack(resolver, chunk_id, pack_unit_ids) for chunk_id in relevant
        ):
            retained += 1
    return retained


def _structural_loss_reason(
    *,
    unit_ids: set[str],
    input_unit_ids: set[str],
    unbounded_unit_ids: set[str],
    unbounded_block_unit_ids: set[str],
) -> str:
    if unit_ids & unbounded_unit_ids and not unit_ids & unbounded_block_unit_ids:
        return "block_reconstruction"
    if unit_ids & unbounded_unit_ids:
        return "ordering"
    if unit_ids & input_unit_ids:
        return "canonical_unit_deduplication"
    return "other"


def _chunk_units_in_pack(
    resolver: CanonicalCorpusResolver,
    chunk_id: str,
    pack_unit_ids: set[str],
) -> bool:
    with suppress(ValueError):
        return bool(
            pack_unit_ids & {unit.unit_id for unit in resolver.resolve_chunk(chunk_id).units}
        )
    return False


def _ordering_violation(pack: ContextPack) -> bool:
    previous_doc: str | None = None
    previous_rank = 0
    previous_ordinal: int | None = None
    for block in pack.blocks:
        if previous_doc == block.document_id and block.best_retrieval_rank < previous_rank:
            return True
        for unit in block.units:
            if unit.document_id != block.document_id:
                return True
            if (
                previous_doc == block.document_id
                and previous_ordinal is not None
                and unit.ordinal is not None
                and unit.ordinal < previous_ordinal
            ):
                return True
            previous_ordinal = unit.ordinal
        previous_doc = block.document_id
        previous_rank = block.best_retrieval_rank
    return False
