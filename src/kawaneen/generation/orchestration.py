"""Model-gated Phase-10 DEV generation plumbing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from kawaneen.evaluation.models import Answerability, DatasetItem, DatasetSplit
from kawaneen.evaluation.serialization import read_items_jsonl
from kawaneen.generation.abstention import abstain, invalid_generation_result
from kawaneen.generation.answerability import (
    ANSWERABILITY_POLICY_VERSION,
    answerability_policy_hash,
    evaluate_stage_d_policy,
    load_source_eligibility_registry,
    load_structural_roles,
)
from kawaneen.generation.artifacts import artifact_fingerprint, write_text_free_artifact
from kawaneen.generation.budgeting import BudgetedContext, budget_context
from kawaneen.generation.checkpoints import (
    GENERATION_CHECKPOINT_ARTIFACT_TYPE,
    GENERATION_CHECKPOINT_SCHEMA_VERSION,
    GenerationCheckpointStore,
    QueryCheckpoint,
)
from kawaneen.generation.context import assemble_or_load_generator_context
from kawaneen.generation.contracts import (
    STAGE_B_GENERATION_SETTINGS,
    STAGE_C_GENERATION_SETTINGS,
    STAGE_D_GENERATION_SETTINGS,
    AbstentionReason,
    GenerationDecision,
    GenerationRequest,
    GenerationResult,
    GenerationSettings,
    TokenizerFingerprint,
    stage_c_generation_payload_schema,
    stage_d_generation_payload_schema,
)
from kawaneen.generation.ollama import (
    LOCAL_OLLAMA_LOCK_PATH,
    OllamaGenerator,
    load_local_model_lock,
)
from kawaneen.generation.policy import (
    PolicyContext,
    PolicyOutcome,
    evaluate_pre_generation_policy,
)
from kawaneen.generation.postprocessing import finalize_generation
from kawaneen.generation.prompt import (
    PROMPT_TEMPLATE_VERSION,
    STAGE_B_PROMPT_TEMPLATE_VERSION,
    STAGE_D_PROMPT_TEMPLATE_VERSION,
    generation_version_hash,
    render_generation_prompt,
    render_stage_b_generation_prompt,
    render_stage_d_generation_prompt,
    stage_b_generation_version_hash,
    stage_c_generation_version_hash,
    stage_d_generation_version_hash,
)
from kawaneen.generation.registry import load_generation_lock
from kawaneen.generation.stage_c import (
    STAGE_C_GENERATOR_NAME,
    stage_c_fingerprint,
)
from kawaneen.generation.stage_c_context import (
    StageCPreparedContext,
    assemble_or_load_stage_c_context,
)
from kawaneen.generation.stage_d import (
    STAGE_D_GENERATOR_NAME,
    STAGE_D_QUOTE_REGISTRY_POLICY_VERSION,
    stage_d_fingerprint,
)
from kawaneen.generation.tokenizer import LazyHuggingFaceTokenizer, TokenizerAdapter
from kawaneen.grounding.assembly import ContextAssembler
from kawaneen.grounding.contracts import ContextPack, RetrievalInput, TokenCounter
from kawaneen.grounding.dev import CANONICAL_DOCUMENTS, CANONICAL_UNITS, CHUNKS, CORPUS_MANIFEST
from kawaneen.grounding.evaluation import audit_dev_contexts, audit_evidence_retention
from kawaneen.grounding.inputs import PHASE8_SELECTION_SHA256, load_frozen_phase8_dev_rankings
from kawaneen.grounding.provenance import CanonicalCorpusResolver

RUNTIME_ITEMS = Path(
    "artifacts/private/phase6_evaluation/ai-reviewed-v1/draft/selected_and_variants.jsonl"
)
PHASE9_POLICY = Path("data/manifests/grounding/phase9_context_policy.json")
GENERATION_PRIVATE_ROOT = Path("artifacts/private/phase10_generation")
QWEN_CONTEXT_CACHE_ROOT = GENERATION_PRIVATE_ROOT / "context_packs" / "qwen-ollama"
DEFAULT_CHECKPOINT_ROOT = GENERATION_PRIVATE_ROOT / "checkpoints" / "qwen-ollama"
DEFAULT_RESULTS_ROOT = GENERATION_PRIVATE_ROOT / "results" / "qwen-ollama"
STAGE_B_GENERATOR_NAME = "qwen-ollama-stage-b"
STAGE_B_CONTEXT_CACHE_ROOT = GENERATION_PRIVATE_ROOT / "context_packs" / STAGE_B_GENERATOR_NAME
STAGE_B_CHECKPOINT_ROOT = GENERATION_PRIVATE_ROOT / "checkpoints" / STAGE_B_GENERATOR_NAME
STAGE_B_RESULTS_ROOT = GENERATION_PRIVATE_ROOT / "results" / STAGE_B_GENERATOR_NAME
STAGE_C_CONTEXT_CACHE_ROOT = (
    GENERATION_PRIVATE_ROOT / "context_packs" / STAGE_C_GENERATOR_NAME
)
STAGE_C_QUOTE_REGISTRY_ROOT = (
    GENERATION_PRIVATE_ROOT / "quote_registries" / STAGE_C_GENERATOR_NAME
)
STAGE_C_CHECKPOINT_ROOT = GENERATION_PRIVATE_ROOT / "checkpoints" / STAGE_C_GENERATOR_NAME
STAGE_C_RESULTS_ROOT = GENERATION_PRIVATE_ROOT / "results" / STAGE_C_GENERATOR_NAME
STAGE_C_READINESS_ROOT = GENERATION_PRIVATE_ROOT / "readiness" / STAGE_C_GENERATOR_NAME
STAGE_C_EXPECTED_QUERY_COUNT = 160
STAGE_D_CONTEXT_CACHE_ROOT = (
    GENERATION_PRIVATE_ROOT / "context_packs" / STAGE_D_GENERATOR_NAME
)
STAGE_D_QUOTE_REGISTRY_ROOT = (
    GENERATION_PRIVATE_ROOT / "quote_registries" / STAGE_D_GENERATOR_NAME
)
STAGE_D_CHECKPOINT_ROOT = GENERATION_PRIVATE_ROOT / "checkpoints" / STAGE_D_GENERATOR_NAME
STAGE_D_RESULTS_ROOT = GENERATION_PRIVATE_ROOT / "results" / STAGE_D_GENERATOR_NAME
STAGE_D_READINESS_ROOT = GENERATION_PRIVATE_ROOT / "readiness" / STAGE_D_GENERATOR_NAME
STAGE_D_EXPECTED_QUERY_COUNT = 160
STAGE_D_TRACKED_READINESS = Path("data/evaluation/phase10_qwen_stage_d_readiness.json")
STAGE_C_UNANSWERABLE_REVIEW = (
    GENERATION_PRIVATE_ROOT / "reviews" / "qwen-ollama-stage-c-unanswerable.json"
)


class RuntimeQuery(BaseModel):
    """The only query record shape allowed into runtime generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)


class RuntimeGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class _TokenizerCounter:
    def __init__(self, tokenizer: TokenizerAdapter) -> None:
        self.tokenizer = tokenizer

    @property
    def identity(self) -> str:
        fingerprint = self.tokenizer.fingerprint
        return ":".join(item for item in (fingerprint.identity, fingerprint.revision) if item)

    def count(self, text: str) -> int:
        return self.tokenizer.count(text)


def load_runtime_dev_queries(path: Path = RUNTIME_ITEMS) -> tuple[RuntimeQuery, ...]:
    """Read only query IDs and query text; qrels never enter this contract."""

    result: list[RuntimeQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("runtime query record is not an object")
        raw = cast(dict[str, object], value)
        split = raw.get("split")
        if split is not None and split != "dev":
            continue
        query_id, query = raw.get("query_id"), raw.get("query_text")
        if not isinstance(query_id, str) or not isinstance(query, str):
            raise ValueError("runtime query record is missing query fields")
        result.append(RuntimeQuery(query_id=query_id, query=query))
    if len({item.query_id for item in result}) != len(result):
        raise ValueError("runtime DEV queries contain duplicate query IDs")
    return tuple(sorted(result, key=lambda item: item.query_id))


def generation_fingerprint(
    *,
    query_id: str,
    context_pack: ContextPack,
    model_revision: str,
    ollama_digest: str,
    tokenizer_fingerprint: TokenizerFingerprint,
    prompt_template_hash: str,
    generation_policy_hash: str,
    phase9_policy_hash: str = "",
    settings: GenerationSettings | None = None,
    generator_name: str = "qwen-ollama",
) -> str:
    effective_settings = settings or GenerationSettings()
    return artifact_fingerprint(
        {
            "generator_name": generator_name,
            "query_id": query_id,
            "phase8_selection_sha256": context_pack.phase8_selection_sha256,
            "phase9_policy_hash": phase9_policy_hash,
            "phase9_policy_version": context_pack.assembly_policy_version,
            "context_pack_hash": artifact_fingerprint(context_pack.model_dump(mode="json")),
            "input_chunk_ids": list(context_pack.input_chunk_ids),
            "qwen_model_revision": model_revision,
            "ollama_digest": ollama_digest,
            "tokenizer_identity": tokenizer_fingerprint.identity,
            "tokenizer_revision": tokenizer_fingerprint.revision,
            "prompt_template_hash": prompt_template_hash,
            "generation_policy_hash": generation_policy_hash,
            "decoding_settings": effective_settings.model_dump(mode="json"),
        }
    )


def generation_status(
    generator_name: str,
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    context_cache_root: Path = STAGE_C_CONTEXT_CACHE_ROOT,
    registry_root: Path = STAGE_C_QUOTE_REGISTRY_ROOT,
) -> dict[str, object]:
    """Return checkpoint metadata without loading model, tokenizer, corpus, or source."""

    if generator_name not in {
        "qwen-ollama",
        STAGE_B_GENERATOR_NAME,
        STAGE_C_GENERATOR_NAME,
        STAGE_D_GENERATOR_NAME,
    }:
        raise ValueError(f"unsupported generation status target: {generator_name}")
    if generator_name == STAGE_B_GENERATOR_NAME and checkpoint_root == DEFAULT_CHECKPOINT_ROOT:
        checkpoint_root = STAGE_B_CHECKPOINT_ROOT
    if generator_name == STAGE_C_GENERATOR_NAME and checkpoint_root == DEFAULT_CHECKPOINT_ROOT:
        checkpoint_root = STAGE_C_CHECKPOINT_ROOT
    if generator_name == STAGE_D_GENERATOR_NAME and checkpoint_root == DEFAULT_CHECKPOINT_ROOT:
        checkpoint_root = STAGE_D_CHECKPOINT_ROOT
    if generator_name == STAGE_C_GENERATOR_NAME:
        counts = GenerationCheckpointStore(
            checkpoint_root, require_complete_lifecycle=True
        ).status()
        contexts_ready = _count_paired_cache_files(context_cache_root)
        registries_ready = _count_paired_cache_files(registry_root)
        return {
            "generator": generator_name,
            "checkpoint_root": checkpoint_root.as_posix(),
            "expected": STAGE_C_EXPECTED_QUERY_COUNT,
            "generation_completed": counts["completed"],
            "generation_missing": max(
                0,
                STAGE_C_EXPECTED_QUERY_COUNT
                - len(tuple(checkpoint_root.glob("*.json")))
            ),
            "contexts_ready": contexts_ready,
            "quote_registries_ready": registries_ready,
            "incomplete": counts["incomplete"],
            "corrupt": counts["corrupt"],
            "model_loaded": False,
            "source_loaded": False,
        }
    if generator_name == STAGE_D_GENERATOR_NAME:
        counts = GenerationCheckpointStore(
            checkpoint_root, require_complete_lifecycle=True
        ).status()
        contexts_ready = _count_paired_cache_files(
            STAGE_D_CONTEXT_CACHE_ROOT
            if context_cache_root == STAGE_C_CONTEXT_CACHE_ROOT
            else context_cache_root
        )
        registries_ready = _count_paired_cache_files(
            STAGE_D_QUOTE_REGISTRY_ROOT
            if registry_root == STAGE_C_QUOTE_REGISTRY_ROOT
            else registry_root
        )
        return {
            "generator": generator_name,
            "checkpoint_root": checkpoint_root.as_posix(),
            "expected": STAGE_D_EXPECTED_QUERY_COUNT,
            "generation_completed": counts["completed"],
            "generation_missing": max(
                0,
                STAGE_D_EXPECTED_QUERY_COUNT - len(tuple(checkpoint_root.glob("*.json"))),
            ),
            "contexts_ready": contexts_ready,
            "quote_registries_ready": registries_ready,
            "incomplete": counts["incomplete"],
            "corrupt": counts["corrupt"],
            "model_loaded": False,
            "source_loaded": False,
        }
    counts = GenerationCheckpointStore(checkpoint_root).status()
    return {
        "generator": generator_name,
        "checkpoint_root": checkpoint_root.as_posix(),
        "completed": counts["completed"],
        "corrupt": counts["corrupt"],
        "model_loaded": False,
        "source_loaded": False,
    }


def generation_readiness(
    generator_name: str,
    *,
    tokenizer: TokenizerAdapter | None = None,
    settings: GenerationSettings | None = None,
    context_cache_root: Path = QWEN_CONTEXT_CACHE_ROOT,
) -> dict[str, object]:
    """Assemble and audit generator contexts without constructing a generator."""

    if generator_name not in {
        "qwen-ollama",
        STAGE_B_GENERATOR_NAME,
        STAGE_C_GENERATOR_NAME,
        STAGE_D_GENERATOR_NAME,
    }:
        raise ValueError(f"unsupported generation readiness target: {generator_name}")
    if generator_name == STAGE_C_GENERATOR_NAME:
        return stage_c_readiness()
    if generator_name == STAGE_D_GENERATOR_NAME:
        return stage_d_readiness()
    stage_b = generator_name == STAGE_B_GENERATOR_NAME
    if stage_b and context_cache_root == QWEN_CONTEXT_CACHE_ROOT:
        context_cache_root = STAGE_B_CONTEXT_CACHE_ROOT
    _, tokenizer_fingerprint = load_generation_lock()
    effective_tokenizer = tokenizer or LazyHuggingFaceTokenizer(
        identity=tokenizer_fingerprint.identity,
        revision=cast(str, tokenizer_fingerprint.revision),
    )
    effective_settings = settings or (
        STAGE_B_GENERATION_SETTINGS if stage_b else GenerationSettings()
    )
    prompt_renderer = render_stage_b_generation_prompt if stage_b else render_generation_prompt
    queries, seeds, factories, resolver = _production_context_inputs(effective_tokenizer)
    phase9_hash = _sha256_file(PHASE9_POLICY) if PHASE9_POLICY.is_file() else ""
    jurisdiction_text = "SA"
    budgeted_contexts: list[BudgetedContext] = []
    assembly_errors = 0
    for runtime_query in queries:
        try:
            budgeted_contexts.append(
                assemble_or_load_generator_context(
                    query=runtime_query.query,
                    context_seed=seeds[runtime_query.query_id],
                    tokenizer=effective_tokenizer,
                    assembler_factory=factories[runtime_query.query_id],
                    settings=effective_settings,
                    phase9_policy_hash=phase9_hash,
                    cache_root=context_cache_root,
                    jurisdiction_text=jurisdiction_text,
                    prompt_renderer=prompt_renderer,
                    prompt_template_version=(
                        STAGE_B_PROMPT_TEMPLATE_VERSION if stage_b else PROMPT_TEMPLATE_VERSION
                    ),
                )
            )
        except (KeyError, OSError, ValueError):
            assembly_errors += 1
    packs = tuple(item.context_pack for item in budgeted_contexts)
    rankings = _group_rankings(load_frozen_phase8_dev_rankings())
    assembly_metrics = audit_dev_contexts(packs, rankings, resolver=resolver)
    prompt_tokens = [item.prompt_token_count for item in budgeted_contexts]
    context_tokens = [item.context_pack.token_count for item in budgeted_contexts]
    budget_violations = sum(
        int(
            item.prompt_token_count > effective_settings.total_input_tokens
            or item.context_pack.token_count > item.evidence_budget_tokens
        )
        for item in budgeted_contexts
    )
    result: dict[str, object] = {
        "status": (
            "qwen_tokenizer_context_readiness_complete"
            if assembly_errors == 0 and len(packs) == len(queries)
            else "qwen_tokenizer_context_readiness_incomplete"
        ),
        "generator": generator_name,
        "query_count": len(queries),
        "assembled_query_count": len(packs),
        "assembly_errors": assembly_errors,
        "tokenizer_id": effective_tokenizer.fingerprint.identity,
        "tokenizer_revision": effective_tokenizer.fingerprint.revision,
        "prompt_tokens": _token_summary(prompt_tokens),
        "context_tokens": _token_summary(context_tokens),
        "budget_violations": budget_violations,
        "mid_unit_truncations": 0,
        "unresolved_provenance": assembly_metrics["unresolved_source_count"],
        "phase8_selection_sha256": PHASE8_SELECTION_SHA256,
        "context_cache_root": context_cache_root.as_posix(),
    }
    if assembly_errors or len(packs) != len(queries):
        return result

    # Qrels are loaded only after all generator-specific contexts are assembled.
    items = tuple(
        item
        for item in read_items_jsonl(RUNTIME_ITEMS)
        if item.split == DatasetSplit.DEV
    )
    unbounded = _assemble_unbounded_contexts(rankings, resolver, effective_tokenizer)
    retention = audit_evidence_retention(
        packs,
        unbounded,
        rankings,
        resolver=resolver,
        items=items,
    )
    budget_only_losses = cast(dict[str, object], retention["BudgetOnlyLosses"])
    result.update(
        {
            "BudgetedGoldEvidenceRetention": _retention_rate(
                retention["budgeted_retained_gold_representations"],
                retention["input_gold_representations"],
            ),
            "BudgetedCompleteGoldEvidenceRetention": _complete_retention_rate(
                items,
                rankings,
                packs,
                resolver,
            ),
            "relevant_representations_excluded_by_real_qwen_budget": budget_only_losses[
                "gold_representation_losses"
            ],
            "posthoc_evidence_retention": retention,
        }
    )
    return result


def run_dev_generation(
    *,
    generator_name: str,
    resume: bool,
    generator: RuntimeGenerator | None = None,
    tokenizer: TokenizerAdapter | None = None,
    runtime_queries: Sequence[RuntimeQuery] | None = None,
    context_packs: Mapping[str, ContextPack] | None = None,
    assembler_factories: Mapping[str, Callable[[int], ContextPack]] | None = None,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    model_revision: str | None = None,
    ollama_digest: str | None = None,
    settings: GenerationSettings | None = None,
    resolver: CanonicalCorpusResolver | None = None,
    context_cache_root: Path = QWEN_CONTEXT_CACHE_ROOT,
) -> dict[str, object]:
    """Run explicitly; any generator failure abstains and never falls back."""

    if generator_name not in {
        "qwen-ollama",
        STAGE_B_GENERATOR_NAME,
        STAGE_C_GENERATOR_NAME,
        STAGE_D_GENERATOR_NAME,
    }:
        raise ValueError("unsupported Qwen DEV generation target")
    if generator_name == STAGE_C_GENERATOR_NAME:
        return _run_stage_c_generation(
            resume=resume,
            generator=generator,
            tokenizer=tokenizer,
            runtime_queries=runtime_queries,
            context_packs=context_packs,
            assembler_factories=assembler_factories,
            checkpoint_root=checkpoint_root,
            results_root=results_root,
            model_revision=model_revision,
            ollama_digest=ollama_digest,
            resolver=resolver,
            context_cache_root=context_cache_root,
        )
    if generator_name == STAGE_D_GENERATOR_NAME:
        return _run_stage_d_generation(
            resume=resume,
            generator=generator,
            tokenizer=tokenizer,
            runtime_queries=runtime_queries,
            context_packs=context_packs,
            assembler_factories=assembler_factories,
            checkpoint_root=checkpoint_root,
            results_root=results_root,
            model_revision=model_revision,
            ollama_digest=ollama_digest,
            resolver=resolver,
            context_cache_root=context_cache_root,
        )
    stage_b = generator_name == STAGE_B_GENERATOR_NAME
    if stage_b:
        if checkpoint_root == DEFAULT_CHECKPOINT_ROOT:
            checkpoint_root = STAGE_B_CHECKPOINT_ROOT
        if results_root == DEFAULT_RESULTS_ROOT:
            results_root = STAGE_B_RESULTS_ROOT
        if context_cache_root == QWEN_CONTEXT_CACHE_ROOT:
            context_cache_root = STAGE_B_CONTEXT_CACHE_ROOT
    effective_settings = settings or (
        STAGE_B_GENERATION_SETTINGS if stage_b else GenerationSettings()
    )
    prompt_renderer = render_stage_b_generation_prompt if stage_b else render_generation_prompt
    if generator is None or tokenizer is None:
        production = _production_runtime(generator_name)
        generator, tokenizer = production[0], production[1]
        runtime_queries = runtime_queries or production[2]
        context_packs = context_packs or production[3]
        assembler_factories = assembler_factories or production[4]
        resolver = resolver or production[5]
        model_revision = model_revision or production[6]
        ollama_digest = ollama_digest or production[7]
    if model_revision is None or ollama_digest is None:
        raise ValueError("Qwen model revision and Ollama digest are required")
    if runtime_queries is None or context_packs is None:
        raise ValueError("runtime DEV inputs are incomplete")
    phase9_hash = _sha256_file(PHASE9_POLICY) if PHASE9_POLICY.is_file() else ""
    prompt_hash = artifact_fingerprint(
        {"version": STAGE_B_PROMPT_TEMPLATE_VERSION if stage_b else PROMPT_TEMPLATE_VERSION}
    )
    policy_hash = (
        stage_b_generation_version_hash(effective_settings)
        if stage_b
        else generation_version_hash(effective_settings)
    )
    checkpoints = GenerationCheckpointStore(checkpoint_root)
    results_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, object] = {
        "total": len(runtime_queries),
        "resumed": 0,
        "answers": 0,
        "abstentions": 0,
    }
    for runtime_query in sorted(runtime_queries, key=lambda item: item.query_id):
        seed_pack = context_packs.get(runtime_query.query_id)
        if seed_pack is None:
            raise ValueError(f"missing frozen Phase-8 ranking: {runtime_query.query_id}")
        assembler_factory = (
            assembler_factories.get(runtime_query.query_id)
            if assembler_factories is not None
            else None
        )
        preparation_error: Exception | None = None
        budgeted: BudgetedContext | None = None
        try:
            if assembler_factory is not None:
                budgeted = assemble_or_load_generator_context(
                    query=runtime_query.query,
                    context_seed=seed_pack,
                    tokenizer=tokenizer,
                    assembler_factory=assembler_factory,
                    settings=effective_settings,
                    phase9_policy_hash=phase9_hash,
                    cache_root=context_cache_root,
                    prompt_renderer=prompt_renderer,
                    prompt_template_version=(
                        STAGE_B_PROMPT_TEMPLATE_VERSION if stage_b else PROMPT_TEMPLATE_VERSION
                    ),
                )
            else:
                budgeted = budget_context(
                    query=runtime_query.query,
                    context_pack=seed_pack,
                    tokenizer=tokenizer,
                    settings=effective_settings,
                    prompt_renderer=prompt_renderer,
                )
            pack = budgeted.context_pack
        except Exception as error:
            preparation_error = error
            pack = seed_pack
        outcome: PolicyOutcome | None = None
        if preparation_error is None:
            outcome = evaluate_pre_generation_policy(
                runtime_query.query,
                PolicyContext(context_pack=pack),
            )
            if outcome.allowed:
                try:
                    if assembler_factory is not None:
                        budgeted = assemble_or_load_generator_context(
                            query=runtime_query.query,
                            context_seed=seed_pack,
                            tokenizer=tokenizer,
                            assembler_factory=assembler_factory,
                            settings=effective_settings,
                            phase9_policy_hash=phase9_hash,
                            cache_root=context_cache_root,
                            jurisdiction_text=outcome.jurisdiction_text,
                            prompt_renderer=prompt_renderer,
                            prompt_template_version=(
                                STAGE_B_PROMPT_TEMPLATE_VERSION
                                if stage_b
                                else PROMPT_TEMPLATE_VERSION
                            ),
                        )
                    else:
                        budgeted = budget_context(
                            query=runtime_query.query,
                            context_pack=pack,
                            tokenizer=tokenizer,
                            settings=effective_settings,
                            jurisdiction_text=outcome.jurisdiction_text,
                            prompt_renderer=prompt_renderer,
                        )
                    pack = budgeted.context_pack
                except Exception as error:
                    preparation_error = error
        fingerprint = generation_fingerprint(
            query_id=runtime_query.query_id,
            context_pack=pack,
            model_revision=model_revision,
            ollama_digest=ollama_digest,
            tokenizer_fingerprint=tokenizer.fingerprint,
            prompt_template_hash=prompt_hash,
            generation_policy_hash=policy_hash,
            phase9_policy_hash=phase9_hash,
            settings=effective_settings,
            generator_name=generator_name,
        )
        result_path = results_root / f"{runtime_query.query_id}.json"
        if resume and result_path.is_file() and checkpoints.valid(
            runtime_query.query_id, fingerprint
        ):
            counts["resumed"] = int(cast(int, counts["resumed"])) + 1
            continue
        raw_output: str | None = None
        rendered_answer: str | None = None
        if preparation_error is not None:
            result = invalid_generation_result(str(preparation_error))
        elif outcome is None:
            result = abstain(AbstentionReason.JURISDICTION_AMBIGUOUS)
        elif not outcome.allowed:
            result = abstain(outcome.reason or AbstentionReason.JURISDICTION_AMBIGUOUS)
        else:
            try:
                if budgeted is None:
                    raise AssertionError("generator context was not budgeted")
                result = generator.generate(
                    GenerationRequest(
                        query=runtime_query.query,
                        context_pack=budgeted.context_pack,
                        settings=effective_settings,
                        jurisdiction_text=outcome.jurisdiction_text,
                    )
                )
                raw_value = getattr(generator, "last_raw_response", None)
                raw_output = raw_value if isinstance(raw_value, str) else None
                if result.decision.value == "answer" and resolver is not None:
                    finalized = finalize_generation(
                        budgeted.context_pack,
                        result,
                        resolver,
                        jurisdiction_text=outcome.jurisdiction_text,
                    )
                    result, rendered_answer = finalized.result, finalized.rendered_answer
            except Exception as error:
                result = invalid_generation_result(str(error))
        _write_private_result(
            result_path,
            {
                "query_id": runtime_query.query_id,
                "fingerprint": fingerprint,
                "raw_output": raw_output,
                "result": result.model_dump(mode="json"),
                "rendered_answer": rendered_answer,
            },
        )
        checkpoints.write(
            QueryCheckpoint(
                query_id=runtime_query.query_id,
                generator_name=generator_name,
                result_path=result_path.as_posix(),
                fingerprint=fingerprint,
            )
        )
        key = "answers" if result.decision.value == "answer" else "abstentions"
        counts[key] = int(cast(int, counts[key])) + 1
    return counts


def _run_stage_c_generation(
    *,
    resume: bool,
    generator: RuntimeGenerator | None,
    tokenizer: TokenizerAdapter | None,
    runtime_queries: Sequence[RuntimeQuery] | None,
    context_packs: Mapping[str, ContextPack] | None,
    assembler_factories: Mapping[str, Callable[[int], ContextPack]] | None,
    checkpoint_root: Path,
    results_root: Path,
    model_revision: str | None,
    ollama_digest: str | None,
    resolver: CanonicalCorpusResolver | None,
    context_cache_root: Path,
) -> dict[str, object]:
    """Run Stage C only in its isolated checkpoint and result namespaces."""

    if checkpoint_root == DEFAULT_CHECKPOINT_ROOT:
        checkpoint_root = STAGE_C_CHECKPOINT_ROOT
    if results_root == DEFAULT_RESULTS_ROOT:
        results_root = STAGE_C_RESULTS_ROOT
    if context_cache_root == QWEN_CONTEXT_CACHE_ROOT:
        context_cache_root = STAGE_C_CONTEXT_CACHE_ROOT
    registry_root = (
        STAGE_C_QUOTE_REGISTRY_ROOT
        if context_cache_root == STAGE_C_CONTEXT_CACHE_ROOT
        else context_cache_root.parent.parent / "quote_registries" / STAGE_C_GENERATOR_NAME
    )
    if generator is None or tokenizer is None or runtime_queries is None or context_packs is None:
        production = _production_stage_c_runtime()
        generator = generator or production[0]
        tokenizer = tokenizer or production[1]
        runtime_queries = runtime_queries or production[2]
        context_packs = context_packs or production[3]
        assembler_factories = assembler_factories or production[4]
        resolver = resolver or production[5]
        model_revision = model_revision or production[6]
        ollama_digest = ollama_digest or production[7]
    if model_revision is None or ollama_digest is None or resolver is None:
        raise ValueError("Stage-C model, digest, and resolver inputs are incomplete")
    assert tokenizer is not None
    assert runtime_queries is not None
    assert context_packs is not None
    assert generator is not None
    if assembler_factories is None:
        assembler_factories = {}
    phase9_hash = _sha256_file(PHASE9_POLICY) if PHASE9_POLICY.is_file() else ""
    prompt_hash = artifact_fingerprint({"version": "phase10-stage-c-prompt-template-v1"})
    schema_hash = artifact_fingerprint(stage_c_generation_payload_schema())
    policy_hash = stage_c_generation_version_hash(STAGE_C_GENERATION_SETTINGS)
    checkpoints = GenerationCheckpointStore(checkpoint_root, require_complete_lifecycle=True)
    results_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, object] = {
        "total": len(runtime_queries),
        "resumed": 0,
        "answers": 0,
        "abstentions": 0,
    }
    for runtime_query in sorted(runtime_queries, key=lambda item: item.query_id):
        seed = context_packs.get(runtime_query.query_id)
        if seed is None:
            raise ValueError(f"missing frozen Phase-8 ranking: {runtime_query.query_id}")
        factory = assembler_factories.get(runtime_query.query_id)
        if factory is None:
            raise ValueError(f"missing Phase-9 provenance assembly input: {runtime_query.query_id}")
        preparation_error: Exception | None = None
        prepared = None
        try:
            prepared = assemble_or_load_stage_c_context(
                query=runtime_query.query,
                context_seed=seed,
                tokenizer=tokenizer,
                assembler_factory=factory,
                settings=STAGE_C_GENERATION_SETTINGS,
                phase9_policy_hash=phase9_hash,
                cache_root=context_cache_root,
                registry_root=registry_root,
                jurisdiction_text=None,
            )
        except Exception as error:
            preparation_error = error
        outcome: PolicyOutcome | None = None
        if preparation_error is None and prepared is not None:
            outcome = evaluate_pre_generation_policy(
                runtime_query.query,
                PolicyContext(context_pack=prepared.context_pack),
            )
            if outcome.allowed:
                try:
                    prepared = assemble_or_load_stage_c_context(
                        query=runtime_query.query,
                        context_seed=seed,
                        tokenizer=tokenizer,
                        assembler_factory=factory,
                        settings=STAGE_C_GENERATION_SETTINGS,
                        phase9_policy_hash=phase9_hash,
                        cache_root=context_cache_root,
                        registry_root=registry_root,
                        jurisdiction_text=outcome.jurisdiction_text,
                    )
                except Exception as error:
                    preparation_error = error
        if prepared is None:
            fingerprint = artifact_fingerprint(
                {
                    "experiment_id": STAGE_C_GENERATOR_NAME,
                    "query_id": runtime_query.query_id,
                    "phase8_selection_sha256": seed.phase8_selection_sha256,
                    "ordered_input_chunk_ids": list(seed.input_chunk_ids),
                    "error": type(preparation_error).__name__ if preparation_error else "missing",
                }
            )
        else:
            fingerprint = stage_c_fingerprint(
                query_id=runtime_query.query_id,
                context_pack=prepared.context_pack,
                registry=prepared.quote_registry,
                model_revision=model_revision,
                ollama_digest=ollama_digest,
                tokenizer_identity=tokenizer.fingerprint.identity,
                tokenizer_revision=tokenizer.fingerprint.revision,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                policy_hash=policy_hash,
            )
        result_path = results_root / f"{runtime_query.query_id}.json"
        if resume and result_path.is_file() and checkpoints.valid(
            runtime_query.query_id, fingerprint
        ):
            counts["resumed"] = int(cast(int, counts["resumed"])) + 1
            continue
        raw_output: str | None = None
        rendered_answer: str | None = None
        telemetry: dict[str, object] = {}
        context_prepared = prepared is not None
        generation_attempted = False
        generation_completed = False
        final_postprocessing_completed = False
        completion_kind: str | None = None
        pre_generation_policy_decision: str | None = None
        if preparation_error is not None or prepared is None:
            result = invalid_generation_result(
                str(preparation_error)
                if preparation_error is not None
                else "missing Stage-C context"
            )
        elif outcome is None:
            result = abstain(AbstentionReason.JURISDICTION_AMBIGUOUS)
        elif not outcome.allowed:
            policy_reason = outcome.reason or AbstentionReason.JURISDICTION_AMBIGUOUS
            result = abstain(policy_reason)
            completion_kind = "pre_generation_policy"
            pre_generation_policy_decision = policy_reason.value
            final_postprocessing_completed = True
        else:
            generation_attempted = True
            try:
                result = generator.generate(
                    GenerationRequest(
                        query=runtime_query.query,
                        context_pack=prepared.context_pack,
                        settings=STAGE_C_GENERATION_SETTINGS,
                        jurisdiction_text=outcome.jurisdiction_text,
                        quote_registry=prepared.quote_registry,
                    )
                )
                raw_value = getattr(generator, "last_raw_response", None)
                raw_output = raw_value if isinstance(raw_value, str) else None
                telemetry_value = getattr(generator, "last_telemetry", {})
                telemetry = (
                    cast(dict[str, object], telemetry_value)
                    if isinstance(telemetry_value, dict)
                    else {}
                )
                generation_completed = isinstance(raw_output, str)
                if generation_completed and result.decision is GenerationDecision.ANSWER:
                    finalized = finalize_generation(
                        prepared.context_pack,
                        result,
                        resolver,
                        jurisdiction_text=outcome.jurisdiction_text,
                    )
                    result, rendered_answer = finalized.result, finalized.rendered_answer
                final_postprocessing_completed = generation_completed
            except Exception as error:
                result = invalid_generation_result(str(error))
                telemetry = {
                    "exception_class": type(error).__name__,
                    "failure_category": "other",
                }
        _write_private_result(
            result_path,
            {
                "artifact_type": "generation_result",
                "schema_version": GENERATION_CHECKPOINT_SCHEMA_VERSION,
                "lifecycle_state": (
                    "complete"
                    if final_postprocessing_completed
                    and (generation_completed or completion_kind == "pre_generation_policy")
                    else "incomplete"
                ),
                "completion_kind": (
                    "generation"
                    if generation_completed
                    else completion_kind
                ),
                "context_prepared": context_prepared,
                "generation_attempted": generation_attempted,
                "generation_completed": generation_completed,
                "final_postprocessing_completed": final_postprocessing_completed,
                "pre_generation_policy_decision": pre_generation_policy_decision,
                "query_id": runtime_query.query_id,
                "fingerprint": fingerprint,
                "raw_output": raw_output,
                "telemetry": telemetry,
                "result": result.model_dump(mode="json"),
                "rendered_answer": rendered_answer,
            },
        )
        checkpoints.write(
            QueryCheckpoint(
                artifact_type=GENERATION_CHECKPOINT_ARTIFACT_TYPE,
                schema_version=GENERATION_CHECKPOINT_SCHEMA_VERSION,
                lifecycle_state=(
                    "complete"
                    if final_postprocessing_completed
                    and (generation_completed or completion_kind == "pre_generation_policy")
                    else "incomplete"
                ),
                completion_kind=("generation" if generation_completed else completion_kind),
                context_prepared=context_prepared,
                generation_attempted=generation_attempted,
                generation_completed=generation_completed,
                final_postprocessing_completed=final_postprocessing_completed,
                pre_generation_policy_decision=pre_generation_policy_decision,
                query_id=runtime_query.query_id,
                generator_name=STAGE_C_GENERATOR_NAME,
                result_path=result_path.as_posix(),
                fingerprint=fingerprint,
                telemetry=telemetry,
            )
        )
        key = "answers" if result.decision is GenerationDecision.ANSWER else "abstentions"
        counts[key] = int(cast(int, counts[key])) + 1
    return counts

def _run_stage_d_generation(
    *,
    resume: bool,
    generator: RuntimeGenerator | None,
    tokenizer: TokenizerAdapter | None,
    runtime_queries: Sequence[RuntimeQuery] | None,
    context_packs: Mapping[str, ContextPack] | None,
    assembler_factories: Mapping[str, Callable[[int], ContextPack]] | None,
    checkpoint_root: Path,
    results_root: Path,
    model_revision: str | None,
    ollama_digest: str | None,
    resolver: CanonicalCorpusResolver | None,
    context_cache_root: Path,
) -> dict[str, object]:
    if checkpoint_root == DEFAULT_CHECKPOINT_ROOT:
        checkpoint_root = STAGE_D_CHECKPOINT_ROOT
    if results_root == DEFAULT_RESULTS_ROOT:
        results_root = STAGE_D_RESULTS_ROOT
    if context_cache_root in {QWEN_CONTEXT_CACHE_ROOT, STAGE_C_CONTEXT_CACHE_ROOT}:
        context_cache_root = STAGE_D_CONTEXT_CACHE_ROOT
    if generator is None or tokenizer is None or runtime_queries is None or context_packs is None:
        production = _production_stage_d_runtime()
        generator = generator or production[0]
        tokenizer = tokenizer or production[1]
        runtime_queries = runtime_queries or production[2]
        context_packs = context_packs or production[3]
        assembler_factories = assembler_factories or production[4]
        resolver = resolver or production[5]
        model_revision = model_revision or production[6]
        ollama_digest = ollama_digest or production[7]
    if model_revision is None or ollama_digest is None or resolver is None:
        raise ValueError("Stage-D model, digest, and resolver inputs are incomplete")
    assert generator is not None and tokenizer is not None and runtime_queries is not None
    assert context_packs is not None
    factories = assembler_factories or {}
    source_registry = load_source_eligibility_registry()
    structural_roles = load_structural_roles()
    phase9_hash = _sha256_file(PHASE9_POLICY) if PHASE9_POLICY.is_file() else ""
    answerability_hash = answerability_policy_hash()
    prompt_hash = artifact_fingerprint({"version": STAGE_D_PROMPT_TEMPLATE_VERSION})
    schema_hash = artifact_fingerprint(stage_d_generation_payload_schema())
    policy_hash = stage_d_generation_version_hash(STAGE_D_GENERATION_SETTINGS)
    checkpoints = GenerationCheckpointStore(checkpoint_root, require_complete_lifecycle=True)
    results_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, object] = {
        "total": len(runtime_queries),
        "resumed": 0,
        "answers": 0,
        "abstentions": 0,
    }
    for runtime_query in sorted(runtime_queries, key=lambda item: item.query_id):
        seed = context_packs.get(runtime_query.query_id)
        factory = factories.get(runtime_query.query_id)
        if seed is None:
            raise ValueError(f"missing frozen Phase-8 ranking: {runtime_query.query_id}")
        if factory is None:
            raise ValueError(f"missing Phase-9 provenance assembly input: {runtime_query.query_id}")
        prepared: StageCPreparedContext | None = None
        preparation_error: Exception | None = None
        try:
            prepared = assemble_or_load_stage_c_context(
                query=runtime_query.query,
                context_seed=seed,
                tokenizer=tokenizer,
                assembler_factory=factory,
                settings=STAGE_D_GENERATION_SETTINGS,
                phase9_policy_hash=phase9_hash,
                cache_root=context_cache_root,
                registry_root=STAGE_D_QUOTE_REGISTRY_ROOT,
                prompt_renderer=render_stage_d_generation_prompt,
                prompt_template_version=STAGE_D_PROMPT_TEMPLATE_VERSION,
                experiment_id=STAGE_D_GENERATOR_NAME,
                quote_registry_policy_version=STAGE_D_QUOTE_REGISTRY_POLICY_VERSION,
            )
        except Exception as error:
            preparation_error = error
        outcome: PolicyOutcome | None = None
        if prepared is not None and preparation_error is None:
            outcome = evaluate_stage_d_policy(
                runtime_query.query,
                PolicyContext(context_pack=prepared.context_pack),
                source_registry=source_registry,
                structural_roles=structural_roles,
            )
            if outcome.allowed:
                try:
                    prepared = assemble_or_load_stage_c_context(
                        query=runtime_query.query,
                        context_seed=seed,
                        tokenizer=tokenizer,
                        assembler_factory=factory,
                        settings=STAGE_D_GENERATION_SETTINGS,
                        phase9_policy_hash=phase9_hash,
                        cache_root=context_cache_root,
                        registry_root=STAGE_D_QUOTE_REGISTRY_ROOT,
                        jurisdiction_text=outcome.jurisdiction_text,
                        prompt_renderer=render_stage_d_generation_prompt,
                        prompt_template_version=STAGE_D_PROMPT_TEMPLATE_VERSION,
                        experiment_id=STAGE_D_GENERATOR_NAME,
                        quote_registry_policy_version=STAGE_D_QUOTE_REGISTRY_POLICY_VERSION,
                    )
                except Exception as error:
                    preparation_error = error
        if prepared is None:
            fingerprint = artifact_fingerprint({
                "experiment_id": STAGE_D_GENERATOR_NAME,
                "query_id": runtime_query.query_id,
                "phase8_selection_sha256": seed.phase8_selection_sha256,
                "ordered_input_chunk_ids": list(seed.input_chunk_ids),
                "answerability_policy_hash": answerability_hash,
                "error": type(preparation_error).__name__ if preparation_error else "missing",
            })
        else:
            fingerprint = stage_d_fingerprint(
                query_id=runtime_query.query_id,
                context_pack=prepared.context_pack,
                registry=prepared.quote_registry,
                model_revision=model_revision,
                ollama_digest=ollama_digest,
                tokenizer_identity=tokenizer.fingerprint.identity,
                tokenizer_revision=tokenizer.fingerprint.revision,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                policy_hash=policy_hash,
                answerability_policy_hash=answerability_hash,
            )
        result_path = results_root / f"{runtime_query.query_id}.json"
        if resume and result_path.is_file() and checkpoints.valid(
            runtime_query.query_id, fingerprint
        ):
            counts["resumed"] = int(cast(int, counts["resumed"])) + 1
            continue
        raw_output: str | None = None
        rendered_answer: str | None = None
        telemetry: dict[str, object] = {}
        context_prepared = prepared is not None
        generation_attempted = False
        generation_completed = False
        final_postprocessing_completed = False
        completion_kind: str | None = None
        policy_decision: str | None = None
        if preparation_error is not None or prepared is None:
            result = invalid_generation_result(
                str(preparation_error) if preparation_error else "missing Stage-D context"
            )
        elif outcome is not None and not outcome.allowed:
            policy_reason = outcome.reason or AbstentionReason.JURISDICTION_AMBIGUOUS
            result = abstain(policy_reason)
            completion_kind = "pre_generation_policy"
            policy_decision = policy_reason.value
            final_postprocessing_completed = True
        elif outcome is None:
            result = abstain(AbstentionReason.JURISDICTION_AMBIGUOUS)
        else:
            generation_attempted = True
            try:
                result = generator.generate(GenerationRequest(
                    query=runtime_query.query,
                    context_pack=prepared.context_pack,
                    settings=STAGE_D_GENERATION_SETTINGS,
                    jurisdiction_text=outcome.jurisdiction_text,
                    quote_registry=prepared.quote_registry,
                ))
                raw_value = getattr(generator, "last_raw_response", None)
                raw_output = raw_value if isinstance(raw_value, str) else None
                telemetry_value = getattr(generator, "last_telemetry", {})
                telemetry = (
                    cast(dict[str, object], telemetry_value)
                    if isinstance(telemetry_value, dict)
                    else {}
                )
                generation_completed = isinstance(raw_output, str)
                if generation_completed and result.decision is GenerationDecision.ANSWER:
                    finalized = finalize_generation(
                        prepared.context_pack,
                        result,
                        resolver,
                        jurisdiction_text=outcome.jurisdiction_text,
                    )
                    result, rendered_answer = finalized.result, finalized.rendered_answer
                final_postprocessing_completed = generation_completed
            except Exception as error:
                result = invalid_generation_result(str(error))
                telemetry = {"exception_class": type(error).__name__, "failure_category": "other"}
        lifecycle_state = (
            "complete"
            if final_postprocessing_completed
            and (generation_completed or completion_kind == "pre_generation_policy")
            else "incomplete"
        )
        payload = {
            "artifact_type": "generation_result",
            "schema_version": GENERATION_CHECKPOINT_SCHEMA_VERSION,
            "lifecycle_state": lifecycle_state,
            "completion_kind": "generation" if generation_completed else completion_kind,
            "context_prepared": context_prepared,
            "generation_attempted": generation_attempted,
            "generation_completed": generation_completed,
            "final_postprocessing_completed": final_postprocessing_completed,
            "pre_generation_policy_decision": policy_decision,
            "query_id": runtime_query.query_id,
            "fingerprint": fingerprint,
            "raw_output": raw_output,
            "telemetry": telemetry,
            "result": result.model_dump(mode="json"),
            "rendered_answer": rendered_answer,
        }
        _write_private_result(result_path, payload)
        checkpoints.write(QueryCheckpoint(
            artifact_type=GENERATION_CHECKPOINT_ARTIFACT_TYPE,
            schema_version=GENERATION_CHECKPOINT_SCHEMA_VERSION,
            lifecycle_state=lifecycle_state,
            completion_kind="generation" if generation_completed else completion_kind,
            context_prepared=context_prepared,
            generation_attempted=generation_attempted,
            generation_completed=generation_completed,
            final_postprocessing_completed=final_postprocessing_completed,
            pre_generation_policy_decision=policy_decision,
            query_id=runtime_query.query_id,
            generator_name=STAGE_D_GENERATOR_NAME,
            result_path=result_path.as_posix(),
            fingerprint=fingerprint,
            telemetry=telemetry,
        ))
        key = "answers" if result.decision is GenerationDecision.ANSWER else "abstentions"
        counts[key] = int(cast(int, counts[key])) + 1
    return counts


def _production_runtime(
    generator_name: str = "qwen-ollama",
) -> tuple[
    RuntimeGenerator,
    TokenizerAdapter,
    tuple[RuntimeQuery, ...],
    dict[str, ContextPack],
    dict[str, Callable[[int], ContextPack]],
    CanonicalCorpusResolver,
    str,
    str,
]:
    candidate, fingerprint = load_generation_lock()
    local_identity = load_local_model_lock(LOCAL_OLLAMA_LOCK_PATH)
    if candidate.ollama_model != local_identity.model:
        raise ValueError("local Ollama model tag does not match Qwen registry")
    tokenizer = LazyHuggingFaceTokenizer(
        identity=fingerprint.identity,
        revision=cast(str, fingerprint.revision),
    )
    generator = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model=local_identity.model,
        immutable_digest=local_identity.digest,
        local_lock_path=LOCAL_OLLAMA_LOCK_PATH,
        stage_b=generator_name == STAGE_B_GENERATOR_NAME,
    )
    queries, packs, factories, resolver = _production_context_inputs(tokenizer)
    return (
        generator,
        tokenizer,
        queries,
        packs,
        factories,
        resolver,
        cast(str, candidate.hf_revision),
        local_identity.digest,
    )


def _production_stage_c_runtime() -> tuple[
    RuntimeGenerator,
    TokenizerAdapter,
    tuple[RuntimeQuery, ...],
    dict[str, ContextPack],
    dict[str, Callable[[int], ContextPack]],
    CanonicalCorpusResolver,
    str,
    str,
]:
    candidate, fingerprint = load_generation_lock()
    local_identity = load_local_model_lock(LOCAL_OLLAMA_LOCK_PATH)
    if candidate.ollama_model != local_identity.model:
        raise ValueError("local Ollama model tag does not match Qwen registry")
    tokenizer = LazyHuggingFaceTokenizer(
        identity=fingerprint.identity,
        revision=cast(str, fingerprint.revision),
    )
    tokenizer.preflight()
    generator = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model=local_identity.model,
        immutable_digest=local_identity.digest,
        local_lock_path=LOCAL_OLLAMA_LOCK_PATH,
        stage_c=True,
    )
    queries, packs, factories, resolver = _production_context_inputs(tokenizer)
    return (
        generator,
        tokenizer,
        queries,
        packs,
        factories,
        resolver,
        cast(str, candidate.hf_revision),
        local_identity.digest,
    )


def _production_stage_d_runtime() -> tuple[
    RuntimeGenerator,
    TokenizerAdapter,
    tuple[RuntimeQuery, ...],
    dict[str, ContextPack],
    dict[str, Callable[[int], ContextPack]],
    CanonicalCorpusResolver,
    str,
    str,
]:
    candidate, fingerprint = load_generation_lock()
    local_identity = load_local_model_lock(LOCAL_OLLAMA_LOCK_PATH)
    if candidate.ollama_model != local_identity.model:
        raise ValueError("local Ollama model tag does not match Qwen registry")
    tokenizer = LazyHuggingFaceTokenizer(
        identity=fingerprint.identity,
        revision=cast(str, fingerprint.revision),
    )
    tokenizer.preflight()
    generator = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model=local_identity.model,
        immutable_digest=local_identity.digest,
        local_lock_path=LOCAL_OLLAMA_LOCK_PATH,
        stage_d=True,
    )
    queries, packs, factories, resolver = _production_context_inputs(tokenizer)
    return (
        generator,
        tokenizer,
        queries,
        packs,
        factories,
        resolver,
        cast(str, candidate.hf_revision),
        local_identity.digest,
    )


def stage_c_readiness() -> dict[str, object]:
    """Assemble all Stage-C contexts and registries without model access."""

    candidate, tokenizer_fingerprint = load_generation_lock()
    tokenizer = LazyHuggingFaceTokenizer(
        identity=tokenizer_fingerprint.identity,
        revision=cast(str, tokenizer_fingerprint.revision),
    )
    tokenizer.preflight()
    queries, seeds, factories, resolver = _production_context_inputs(tokenizer)
    phase9_hash = _sha256_file(PHASE9_POLICY) if PHASE9_POLICY.is_file() else ""
    prepared: list[StageCPreparedContext] = []
    errors = 0
    for runtime_query in queries:
        try:
            prepared.append(
                assemble_or_load_stage_c_context(
                    query=runtime_query.query,
                    context_seed=seeds[runtime_query.query_id],
                    tokenizer=tokenizer,
                    assembler_factory=factories[runtime_query.query_id],
                    settings=STAGE_C_GENERATION_SETTINGS,
                    phase9_policy_hash=phase9_hash,
                    cache_root=STAGE_C_CONTEXT_CACHE_ROOT,
                    registry_root=STAGE_C_QUOTE_REGISTRY_ROOT,
                    jurisdiction_text="SA",
                )
            )
        except (KeyError, OSError, ValueError):
            errors += 1
    prepared_contexts = tuple(prepared)
    packs = tuple(item.context_pack for item in prepared_contexts)
    registries = tuple(item.quote_registry for item in prepared_contexts)
    rankings = _group_rankings(load_frozen_phase8_dev_rankings())
    assembly_metrics = audit_dev_contexts(packs, rankings, resolver=resolver)
    prompt_tokens = [item.prompt_token_count for item in prepared_contexts]
    context_tokens = [item.evidence_token_count for item in prepared_contexts]
    registry_counts = [len(item.entries) for item in registries]
    result: dict[str, object] = {
        "status": (
            "qwen_stage_c_context_readiness_complete"
            if errors == 0 and len(prepared_contexts) == len(queries)
            else "qwen_stage_c_context_readiness_incomplete"
        ),
        "generator": STAGE_C_GENERATOR_NAME,
        "query_count": len(queries),
        "assembled_query_count": len(prepared_contexts),
        "valid_contexts": len(prepared_contexts),
        "assembly_errors": errors,
        "tokenizer_id": tokenizer.fingerprint.identity,
        "tokenizer_revision": tokenizer.fingerprint.revision,
        "model_revision": candidate.hf_revision,
        "quote_registry_entries": _token_summary(registry_counts),
        "prompt_tokens": _token_summary(prompt_tokens),
        "context_tokens": _token_summary(context_tokens),
        "budget_violations": sum(
            item.prompt_token_count > STAGE_C_GENERATION_SETTINGS.total_input_tokens
            for item in prepared_contexts
        ),
        "mid_unit_truncations": 0,
        "unresolved_provenance": assembly_metrics["unresolved_source_count"],
        "phase8_selection_sha256": PHASE8_SELECTION_SHA256,
        "context_cache_root": STAGE_C_CONTEXT_CACHE_ROOT.as_posix(),
        "quote_registry_root": STAGE_C_QUOTE_REGISTRY_ROOT.as_posix(),
        "readiness_root": STAGE_C_READINESS_ROOT.as_posix(),
        "timeout_seconds": 60.0,
    }
    if errors or len(prepared_contexts) != len(queries):
        return result
    items = tuple(
        item for item in read_items_jsonl(RUNTIME_ITEMS) if item.split == DatasetSplit.DEV
    )
    unbounded = _assemble_unbounded_contexts(rankings, resolver, tokenizer)
    retention = audit_evidence_retention(
        packs,
        unbounded,
        rankings,
        resolver=resolver,
        items=items,
    )
    budget_only_losses = cast(dict[str, object], retention["BudgetOnlyLosses"])
    result.update(
        {
            "BudgetedGoldEvidenceRetention": _retention_rate(
                retention["budgeted_retained_gold_representations"],
                retention["input_gold_representations"],
            ),
            "BudgetedCompleteGoldEvidenceRetention": _complete_retention_rate(
                items, rankings, packs, resolver
            ),
            "relevant_representations_excluded_by_real_qwen_budget": budget_only_losses[
                "gold_representation_losses"
            ],
            "posthoc_evidence_retention": retention,
        }
    )
    _write_private_result(
        STAGE_C_READINESS_ROOT / "readiness.json",
        {
            "artifact_type": "generation_readiness",
            "schema_version": 1,
            "lifecycle_state": "context_prepared",
            "generator": STAGE_C_GENERATOR_NAME,
            "query_count": len(queries),
            "assembled_query_count": len(prepared_contexts),
            "context_cache_root": STAGE_C_CONTEXT_CACHE_ROOT.as_posix(),
            "quote_registry_root": STAGE_C_QUOTE_REGISTRY_ROOT.as_posix(),
        },
    )
    return result


def stage_d_readiness() -> dict[str, object]:
    """Build Stage-D contexts and apply answerability policy without inference."""

    candidate, tokenizer_fingerprint = load_generation_lock()
    tokenizer = LazyHuggingFaceTokenizer(
        identity=tokenizer_fingerprint.identity,
        revision=cast(str, tokenizer_fingerprint.revision),
    )
    tokenizer.preflight()
    queries, seeds, factories, resolver = _production_context_inputs(tokenizer)
    source_registry = load_source_eligibility_registry()
    structural_roles = load_structural_roles()
    phase9_hash = _sha256_file(PHASE9_POLICY) if PHASE9_POLICY.is_file() else ""
    prepared: list[StageCPreparedContext] = []
    prepared_by_id: dict[str, StageCPreparedContext] = {}
    policy_reasons: defaultdict[str, int] = defaultdict(int)
    model_eligible = 0
    errors = 0
    for runtime_query in queries:
        try:
            item = assemble_or_load_stage_c_context(
                query=runtime_query.query,
                context_seed=seeds[runtime_query.query_id],
                tokenizer=tokenizer,
                assembler_factory=factories[runtime_query.query_id],
                settings=STAGE_D_GENERATION_SETTINGS,
                phase9_policy_hash=phase9_hash,
                cache_root=STAGE_D_CONTEXT_CACHE_ROOT,
                registry_root=STAGE_D_QUOTE_REGISTRY_ROOT,
                jurisdiction_text=None,
                prompt_renderer=render_stage_d_generation_prompt,
                prompt_template_version=STAGE_D_PROMPT_TEMPLATE_VERSION,
                experiment_id=STAGE_D_GENERATOR_NAME,
                quote_registry_policy_version=STAGE_D_QUOTE_REGISTRY_POLICY_VERSION,
            )
            prepared.append(item)
            prepared_by_id[runtime_query.query_id] = item
            outcome = evaluate_stage_d_policy(
                runtime_query.query,
                PolicyContext(context_pack=item.context_pack),
                source_registry=source_registry,
                structural_roles=structural_roles,
            )
            if outcome.allowed:
                model_eligible += 1
            else:
                policy_reasons[
                    (outcome.reason or AbstentionReason.JURISDICTION_AMBIGUOUS).value
                ] += 1
        except (KeyError, OSError, ValueError):
            errors += 1
    packs = tuple(item.context_pack for item in prepared)
    registries = tuple(item.quote_registry for item in prepared)
    rankings = _group_rankings(load_frozen_phase8_dev_rankings())
    assembly_metrics = audit_dev_contexts(packs, rankings, resolver=resolver)
    eligible_ids: set[str] = set()
    for query in queries:
        item = prepared_by_id.get(query.query_id)
        if item is not None and evaluate_stage_d_policy(
            query.query,
            PolicyContext(context_pack=item.context_pack),
            source_registry=source_registry,
            structural_roles=structural_roles,
        ).allowed:
            eligible_ids.add(query.query_id)
    prompt_tokens = [
        item.prompt_token_count
        for query in queries
        for item in (prepared_by_id.get(query.query_id),)
        if item is not None
        if query.query_id in eligible_ids
    ]
    evidence_tokens = [item.evidence_token_count for item in prepared]
    result: dict[str, object] = {
        "status": (
            "qwen_stage_d_readiness_complete"
            if errors == 0 and len(prepared) == len(queries)
            else "qwen_stage_d_readiness_incomplete"
        ),
        "generator": STAGE_D_GENERATOR_NAME,
        "query_count": len(queries),
        "valid_contexts": len(prepared),
        "valid_quote_registries": len(registries),
        "assembly_errors": errors,
        "policy_pre_abstained": sum(policy_reasons.values()),
        "policy_pre_abstained_by_reason": dict(sorted(policy_reasons.items())),
        "model_eligible": model_eligible,
        "tokenizer_id": tokenizer.fingerprint.identity,
        "tokenizer_revision": tokenizer.fingerprint.revision,
        "model_revision": candidate.hf_revision,
        "quote_registry_entries": _token_summary([len(item.entries) for item in registries]),
        "prompt_tokens_model_eligible": _token_summary(prompt_tokens),
        "context_tokens": _token_summary(evidence_tokens),
        "budget_violations": sum(
            item.prompt_token_count > STAGE_D_GENERATION_SETTINGS.total_input_tokens
            for item in prepared
        ),
        "mid_unit_truncations": 0,
        "unresolved_provenance": assembly_metrics["unresolved_source_count"],
        "phase8_selection_sha256": PHASE8_SELECTION_SHA256,
        "context_cache_root": STAGE_D_CONTEXT_CACHE_ROOT.as_posix(),
        "quote_registry_root": STAGE_D_QUOTE_REGISTRY_ROOT.as_posix(),
        "readiness_root": STAGE_D_READINESS_ROOT.as_posix(),
        "timeout_seconds": 60.0,
        "input_cap": STAGE_D_GENERATION_SETTINGS.total_input_tokens,
        "output_cap": STAGE_D_GENERATION_SETTINGS.max_new_tokens,
        "answerability_policy_version": ANSWERABILITY_POLICY_VERSION,
        "answerability_policy_hash": answerability_policy_hash(),
    }
    write_text_free_artifact(
        STAGE_D_TRACKED_READINESS,
        {
            "artifact_type": "stage_d_readiness_summary",
            "schema_version": 1,
            "generator": STAGE_D_GENERATOR_NAME,
            "query_count": len(queries),
            "valid_contexts": len(prepared),
            "valid_quote_registries": len(registries),
            "policy_pre_abstained": sum(policy_reasons.values()),
            "policy_pre_abstained_by_reason": dict(sorted(policy_reasons.items())),
            "model_eligible": model_eligible,
            "prompt_tokens_model_eligible": _token_summary(prompt_tokens),
            "context_tokens": _token_summary(evidence_tokens),
            "budget_violations": result["budget_violations"],
            "mid_unit_truncations": 0,
            "unresolved_provenance": assembly_metrics["unresolved_source_count"],
            "tokenizer_id": tokenizer.fingerprint.identity,
            "tokenizer_revision": tokenizer.fingerprint.revision,
            "model_revision": candidate.hf_revision,
            "input_cap": STAGE_D_GENERATION_SETTINGS.total_input_tokens,
            "output_cap": STAGE_D_GENERATION_SETTINGS.max_new_tokens,
            "timeout_seconds": 60.0,
            "policy_version": ANSWERABILITY_POLICY_VERSION,
            "policy_hash": answerability_policy_hash(),
        },
    )
    _write_private_result(
        STAGE_D_READINESS_ROOT / "readiness.json",
        {
            "artifact_type": "generation_readiness",
            "schema_version": 1,
            "lifecycle_state": "context_prepared",
            "generator": STAGE_D_GENERATOR_NAME,
            "query_count": len(queries),
            "valid_contexts": len(prepared),
            "valid_quote_registries": len(registries),
            "model_eligible": model_eligible,
            "policy_pre_abstained_by_reason": dict(sorted(policy_reasons.items())),
            "policy_hash": answerability_policy_hash(),
        },
    )
    if STAGE_C_UNANSWERABLE_REVIEW.is_file():
        try:
            review_value = json.loads(STAGE_C_UNANSWERABLE_REVIEW.read_text(encoding="utf-8"))
            review = cast(dict[str, object], review_value) if isinstance(review_value, dict) else {}
            cases_value = review.get("cases", [])
            cases = cast(list[object], cases_value) if isinstance(cases_value, list) else []
            audit: list[dict[str, object]] = []
            for case_value in cases:
                case = cast(dict[str, object], case_value) if isinstance(case_value, dict) else {}
                if not isinstance(case.get("query_id"), str):
                    continue
                query_id = cast(str, case["query_id"])
                runtime_query = next((item for item in queries if item.query_id == query_id), None)
                prepared_item = prepared_by_id.get(query_id)
                if runtime_query is None or prepared_item is None:
                    continue
                outcome = evaluate_stage_d_policy(
                    runtime_query.query,
                    PolicyContext(context_pack=prepared_item.context_pack),
                    source_registry=source_registry,
                    structural_roles=structural_roles,
                )
                audit.append(
                    {
                        "query_id_hash": artifact_fingerprint(query_id),
                        "policy_decision": "allow" if outcome.allowed else "abstain",
                        "abstention_reason": outcome.reason.value if outcome.reason else None,
                    }
                )
            _write_private_result(
                STAGE_D_READINESS_ROOT / "stage-c-unanswerable-policy-audit.json",
                {
                    "artifact_type": "stage_d_policy_audit",
                    "schema_version": 1,
                    "case_count": len(audit),
                    "cases": audit,
                },
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return result


def _production_context_inputs(
    tokenizer: TokenizerAdapter,
) -> tuple[
    tuple[RuntimeQuery, ...],
    dict[str, ContextPack],
    dict[str, Callable[[int], ContextPack]],
    CanonicalCorpusResolver,
]:
    rankings = _group_rankings(load_frozen_phase8_dev_rankings())
    queries = load_runtime_dev_queries()
    query_ids = {item.query_id for item in queries}
    ranking_ids = set(rankings)
    if query_ids != ranking_ids:
        missing = sorted(query_ids - ranking_ids)
        extra = sorted(ranking_ids - query_ids)
        raise ValueError(f"Phase-8/runtime query mismatch: missing={missing}, extra={extra}")
    resolver = CanonicalCorpusResolver.from_json(
        CANONICAL_UNITS,
        CHUNKS,
        CORPUS_MANIFEST,
        document_paths=CANONICAL_DOCUMENTS,
    )
    counter = _TokenizerCounter(tokenizer)
    corpus_hash = resolver.corpus_hash
    if corpus_hash is None:
        raise ValueError("canonical corpus hash is unavailable")
    packs = {
        query_id: _empty_context_pack(
            query_id=query_id,
            ranked_inputs=rankings[query_id],
            tokenizer=tokenizer,
            canonical_corpus_hash=corpus_hash,
            chunk_policy_hash=resolver.chunk_policy_hash or "",
        )
        for query_id in rankings
    }
    factories = {
        query_id: _assembler_factory(
            query_id=query_id,
            ranked_inputs=rankings[query_id],
            resolver=resolver,
            counter=counter,
            canonical_corpus_hash=corpus_hash,
        )
        for query_id in rankings
    }
    return queries, packs, factories, resolver


def _assembler_factory(
    *,
    query_id: str,
    ranked_inputs: Sequence[RetrievalInput],
    resolver: CanonicalCorpusResolver,
    counter: TokenCounter,
    canonical_corpus_hash: str,
) -> Callable[[int], ContextPack]:
    def assemble(max_tokens: int) -> ContextPack:
        return ContextAssembler(resolver, counter, max_context_tokens=max_tokens).assemble(
            query_id=query_id,
            ranked_inputs=ranked_inputs,
            phase8_selection_sha256=PHASE8_SELECTION_SHA256,
            canonical_corpus_hash=canonical_corpus_hash,
        )

    return assemble


def _group_rankings(rows: Sequence[RetrievalInput]) -> dict[str, tuple[RetrievalInput, ...]]:
    grouped: defaultdict[str, list[RetrievalInput]] = defaultdict(list)
    for row in rows:
        grouped[row.query_id].append(row)
    return {query_id: tuple(values) for query_id, values in grouped.items()}


def _empty_context_pack(
    *,
    query_id: str,
    ranked_inputs: Sequence[RetrievalInput],
    tokenizer: TokenizerAdapter,
    canonical_corpus_hash: str,
    chunk_policy_hash: str,
) -> ContextPack:
    return ContextPack(
        query_id=query_id,
        phase8_selection_sha256=PHASE8_SELECTION_SHA256,
        canonical_corpus_hash=canonical_corpus_hash,
        assembly_policy_version="phase9-context-assembly-v1",
        token_counter_identity=_TokenizerCounter(tokenizer).identity,
        max_context_tokens=0,
        token_count=0,
        units=(),
        blocks=(),
        evidence=(),
        omissions=(),
        input_chunk_ids=tuple(
            item.chunk_id for item in sorted(ranked_inputs, key=lambda item: item.rank)
        ),
        chunk_policy_hash=chunk_policy_hash,
    )


def _assemble_unbounded_contexts(
    rankings: Mapping[str, Sequence[RetrievalInput]],
    resolver: CanonicalCorpusResolver,
    tokenizer: TokenizerAdapter,
) -> tuple[ContextPack, ...]:
    assembler = ContextAssembler(
        resolver,
        _TokenizerCounter(tokenizer),
        max_context_tokens=2**63 - 1,
    )
    if resolver.corpus_hash is None:
        raise ValueError("canonical corpus hash is unavailable")
    return tuple(
        assembler.assemble(
            query_id=query_id,
            ranked_inputs=rankings[query_id],
            phase8_selection_sha256=PHASE8_SELECTION_SHA256,
            canonical_corpus_hash=resolver.corpus_hash,
        )
        for query_id in sorted(rankings)
    )


def _token_summary(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "p50": _nearest_rank_percentile(ordered, 50),
        "p95": _nearest_rank_percentile(ordered, 95),
        "max": ordered[-1],
    }


def _nearest_rank_percentile(ordered: Sequence[int], percentile: int) -> int:
    index = max(0, math.ceil(percentile * len(ordered) / 100) - 1)
    return ordered[index]


def _retention_rate(retained: object, total: object) -> float:
    retained_count = int(cast(int, retained))
    total_count = int(cast(int, total))
    return retained_count / total_count if total_count else 0.0


def _complete_retention_rate(
    items: Sequence[DatasetItem],
    rankings: Mapping[str, Sequence[RetrievalInput]],
    packs: Sequence[ContextPack],
    resolver: CanonicalCorpusResolver,
) -> float:
    packs_by_query = {pack.query_id: pack for pack in packs}
    input_complete = 0
    budgeted_complete = 0
    for raw_item in items:
        item = raw_item
        if item.answerability != Answerability.ANSWERABLE:
            continue
        query_id = str(item.query_id)
        input_unit_ids = {
            unit.unit_id
            for row in rankings.get(query_id, ())
            for unit in resolver.resolve_chunk(row.chunk_id).units
        }
        groups = item.evidence_groups
        if not all(
            any(span.unit_id in input_unit_ids for span in group.spans) for group in groups
        ):
            continue
        input_complete += 1
        pack_units = {unit.unit_id for unit in packs_by_query[query_id].units}
        budgeted_complete += int(
            all(any(span.unit_id in pack_units for span in group.spans) for group in groups)
        )
    return budgeted_complete / input_complete if input_complete else 0.0


def _write_private_result(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _count_paired_cache_files(root: Path) -> int:
    """Count ready query cache pairs without loading models, corpus, or source text."""

    if not root.is_dir():
        return 0
    return len({path.stem for path in root.glob("*.json")})


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
