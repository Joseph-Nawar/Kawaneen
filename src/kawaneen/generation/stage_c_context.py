"""Tokenizer-budgeted Stage-C ContextPack and QuoteRegistry assembly."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kawaneen.generation.artifacts import artifact_fingerprint
from kawaneen.generation.contracts import GenerationSettings, TokenizerFingerprint
from kawaneen.generation.prompt import (
    STAGE_C_PROMPT_TEMPLATE_VERSION,
    RenderedPrompt,
    render_stage_c_generation_prompt,
)
from kawaneen.generation.quote_registry import (
    QUOTE_REGISTRY_POLICY_VERSION,
    QuoteRegistry,
    build_quote_registry,
    render_quote_registry_context,
)
from kawaneen.generation.stage_c import STAGE_C_GENERATOR_NAME
from kawaneen.generation.tokenizer import TokenizerAdapter
from kawaneen.grounding.contracts import ContextPack

STAGE_C_CONTEXT_CACHE_SCHEMA_VERSION = 1
PromptRenderer = Callable[..., RenderedPrompt]


@dataclass(frozen=True, slots=True)
class StageCPreparedContext:
    context_pack: ContextPack
    quote_registry: QuoteRegistry
    non_evidence_prompt_tokens: int
    evidence_token_count: int
    prompt_token_count: int
    evidence_budget_tokens: int
    omitted_unit_ids: tuple[str, ...]
    fingerprint: str


def stage_c_context_cache_fingerprint(
    *,
    query: str,
    context_seed: ContextPack,
    tokenizer: TokenizerAdapter,
    settings: GenerationSettings,
    phase9_policy_hash: str,
    fixed_prompt_tokens: int,
    evidence_budget_tokens: int,
    jurisdiction_text: str | None,
    experiment_id: str = STAGE_C_GENERATOR_NAME,
    prompt_template_version: str = STAGE_C_PROMPT_TEMPLATE_VERSION,
    quote_registry_policy_version: str = QUOTE_REGISTRY_POLICY_VERSION,
) -> str:
    fingerprint = TokenizerFingerprint(
        identity=tokenizer.fingerprint.identity,
        revision=tokenizer.fingerprint.revision,
        vocabulary_hash=tokenizer.fingerprint.vocabulary_hash,
    )
    return artifact_fingerprint(
        {
            "experiment_id": experiment_id,
            "query_hash": artifact_fingerprint(query),
            "phase8_selection_sha256": context_seed.phase8_selection_sha256,
            "ordered_input_chunk_ids": list(context_seed.input_chunk_ids),
            "phase9_policy_hash": phase9_policy_hash,
            "phase9_assembly_policy": context_seed.assembly_policy_version,
            "quote_registry_policy_version": quote_registry_policy_version,
            "tokenizer_id": fingerprint.identity,
            "tokenizer_revision": fingerprint.revision,
            "prompt_template_version": prompt_template_version,
            "prompt_input_token_cap": settings.total_input_tokens,
            "output_reservation": settings.output_reservation,
            "safety_margin": settings.safety_margin,
            "fixed_prompt_tokens": fixed_prompt_tokens,
            "evidence_token_budget": evidence_budget_tokens,
            "jurisdiction_text": jurisdiction_text,
        }
    )


def assemble_or_load_stage_c_context(
    *,
    query: str,
    context_seed: ContextPack,
    tokenizer: TokenizerAdapter,
    assembler_factory: Callable[[int], ContextPack],
    settings: GenerationSettings,
    phase9_policy_hash: str,
    cache_root: Path,
    registry_root: Path,
    jurisdiction_text: str | None = None,
    prompt_renderer: PromptRenderer = render_stage_c_generation_prompt,
    prompt_template_version: str = STAGE_C_PROMPT_TEMPLATE_VERSION,
    experiment_id: str = STAGE_C_GENERATOR_NAME,
    quote_registry_policy_version: str = QUOTE_REGISTRY_POLICY_VERSION,
) -> StageCPreparedContext:
    empty_pack = context_seed.model_copy(
        update={"units": (), "blocks": (), "evidence": (), "omissions": (), "token_count": 0}
    )
    empty_registry = QuoteRegistry(
        query_id=context_seed.query_id,
        policy_version=quote_registry_policy_version,
        entries=(),
    )
    fixed_prompt_tokens = tokenizer.count(
        prompt_renderer(
            query,
            empty_pack,
            registry=empty_registry,
            settings=settings,
            jurisdiction_text=jurisdiction_text,
        ).text
    )
    evidence_budget = settings.total_input_tokens - fixed_prompt_tokens - settings.safety_margin
    if evidence_budget <= 0:
        raise ValueError("Stage-C fixed prompt overhead leaves no evidence budget")
    fingerprint = stage_c_context_cache_fingerprint(
        query=query,
        context_seed=context_seed,
        tokenizer=tokenizer,
        settings=settings,
        phase9_policy_hash=phase9_policy_hash,
        fixed_prompt_tokens=fixed_prompt_tokens,
        evidence_budget_tokens=evidence_budget,
        jurisdiction_text=jurisdiction_text,
        experiment_id=experiment_id,
        prompt_template_version=prompt_template_version,
        quote_registry_policy_version=quote_registry_policy_version,
    )
    cached = _load_cached(
        cache_root / f"{context_seed.query_id}.json",
        registry_root / f"{context_seed.query_id}.json",
        fingerprint=fingerprint,
        query=query,
        context_seed=context_seed,
        tokenizer=tokenizer,
        settings=settings,
        fixed_prompt_tokens=fixed_prompt_tokens,
        evidence_budget=evidence_budget,
        phase9_policy_hash=phase9_policy_hash,
        jurisdiction_text=jurisdiction_text,
        prompt_renderer=prompt_renderer,
        prompt_template_version=prompt_template_version,
        quote_registry_policy_version=quote_registry_policy_version,
    )
    if cached is not None:
        return cached

    candidate_budget = evidence_budget
    for _ in range(64):
        pack = assembler_factory(candidate_budget)
        registry = build_quote_registry(pack, policy_version=quote_registry_policy_version)
        rendered = prompt_renderer(
            query,
            pack,
            registry=registry,
            settings=settings,
            jurisdiction_text=jurisdiction_text,
        )
        prompt_tokens = tokenizer.count(rendered.text)
        if prompt_tokens <= settings.total_input_tokens:
            evidence_tokens = tokenizer.count(render_quote_registry_context(pack, registry))
            prepared = StageCPreparedContext(
                context_pack=pack,
                quote_registry=registry,
                non_evidence_prompt_tokens=fixed_prompt_tokens,
                evidence_token_count=evidence_tokens,
                prompt_token_count=prompt_tokens,
                evidence_budget_tokens=candidate_budget,
                omitted_unit_ids=tuple(item.unit_id for item in pack.omissions),
                fingerprint=fingerprint,
            )
            _write_json(
                cache_root / f"{context_seed.query_id}.json",
                {
                    "schema_version": STAGE_C_CONTEXT_CACHE_SCHEMA_VERSION,
                    "fingerprint": fingerprint,
                    "query_id": context_seed.query_id,
                    "phase8_selection_sha256": context_seed.phase8_selection_sha256,
                    "ordered_input_chunk_ids": list(context_seed.input_chunk_ids),
                    "phase9_policy_hash": phase9_policy_hash,
                    "quote_registry_policy_version": registry.policy_version,
                    "tokenizer_id": tokenizer.fingerprint.identity,
                    "tokenizer_revision": tokenizer.fingerprint.revision,
                    "prompt_input_token_cap": settings.total_input_tokens,
                    "output_reservation": settings.output_reservation,
                    "safety_margin": settings.safety_margin,
                    "fixed_prompt_tokens": fixed_prompt_tokens,
                    "evidence_token_budget": evidence_budget,
                    "resolved_evidence_token_budget": candidate_budget,
                    "prompt_token_count": prompt_tokens,
                    "evidence_token_count": evidence_tokens,
                    "context_pack": pack.model_dump(mode="json"),
                },
            )
            _write_json(
                registry_root / f"{context_seed.query_id}.json",
                {
                    "schema_version": STAGE_C_CONTEXT_CACHE_SCHEMA_VERSION,
                    "fingerprint": fingerprint,
                    "quote_registry": registry.model_dump(mode="json"),
                },
            )
            return prepared
        candidate_budget -= max(1, prompt_tokens - settings.total_input_tokens)
        if candidate_budget < 0:
            break
    raise ValueError("Stage-C context cannot satisfy the tokenizer input budget")


def _load_cached(
    context_path: Path,
    registry_path: Path,
    *,
    fingerprint: str,
    query: str,
    context_seed: ContextPack,
    tokenizer: TokenizerAdapter,
    settings: GenerationSettings,
    fixed_prompt_tokens: int,
    evidence_budget: int,
    phase9_policy_hash: str,
    jurisdiction_text: str | None,
    prompt_renderer: PromptRenderer,
    prompt_template_version: str,
    quote_registry_policy_version: str,
) -> StageCPreparedContext | None:
    try:
        context_value = json.loads(context_path.read_text(encoding="utf-8"))
        registry_value = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(context_value, dict) or not isinstance(registry_value, dict):
            return None
        context_value = cast(dict[str, object], context_value)
        registry_value = cast(dict[str, object], registry_value)
        chunk_ids_value = context_value.get("ordered_input_chunk_ids", ())
        if not isinstance(chunk_ids_value, list):
            return None
        chunk_ids = tuple(str(item) for item in cast(list[object], chunk_ids_value))
        if (
            context_value.get("schema_version") != STAGE_C_CONTEXT_CACHE_SCHEMA_VERSION
            or context_value.get("fingerprint") != fingerprint
            or registry_value.get("schema_version") != STAGE_C_CONTEXT_CACHE_SCHEMA_VERSION
            or registry_value.get("fingerprint") != fingerprint
            or context_value.get("query_id") != context_seed.query_id
            or context_value.get("phase8_selection_sha256") != context_seed.phase8_selection_sha256
            or chunk_ids != context_seed.input_chunk_ids
            or context_value.get("phase9_policy_hash") != phase9_policy_hash
            or context_value.get("tokenizer_id") != tokenizer.fingerprint.identity
            or context_value.get("tokenizer_revision") != tokenizer.fingerprint.revision
            or context_value.get("prompt_input_token_cap") != settings.total_input_tokens
            or context_value.get("output_reservation") != settings.output_reservation
            or context_value.get("safety_margin") != settings.safety_margin
            or context_value.get("fixed_prompt_tokens") != fixed_prompt_tokens
            or context_value.get("evidence_token_budget") != evidence_budget
        ):
            return None
        pack = ContextPack.model_validate(context_value["context_pack"])
        registry = QuoteRegistry.model_validate(registry_value["quote_registry"])
        if (
            registry.query_id != pack.query_id
            or registry.fingerprint
            != build_quote_registry(pack, policy_version=quote_registry_policy_version).fingerprint
            or pack.token_count > pack.max_context_tokens
        ):
            return None
        rendered = prompt_renderer(
            query,
            pack,
            registry=registry,
            settings=settings,
            jurisdiction_text=jurisdiction_text,
        )
        prompt_tokens = tokenizer.count(rendered.text)
        if prompt_tokens > settings.total_input_tokens:
            return None
        resolved_budget = context_value.get("resolved_evidence_token_budget")
        if not isinstance(resolved_budget, int) or resolved_budget != evidence_budget:
            return None
        evidence_tokens = tokenizer.count(render_quote_registry_context(pack, registry))
        return StageCPreparedContext(
            context_pack=pack,
            quote_registry=registry,
            non_evidence_prompt_tokens=fixed_prompt_tokens,
            evidence_token_count=evidence_tokens,
            prompt_token_count=prompt_tokens,
            evidence_budget_tokens=resolved_budget,
            omitted_unit_ids=tuple(item.unit_id for item in pack.omissions),
            fingerprint=fingerprint,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
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
