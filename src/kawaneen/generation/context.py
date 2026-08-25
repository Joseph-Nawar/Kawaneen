"""Private generator-specific ContextPack assembly and caching."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from kawaneen.generation.artifacts import artifact_fingerprint
from kawaneen.generation.budgeting import (
    BudgetedContext,
    budget_context,
    calculate_evidence_budget,
)
from kawaneen.generation.contracts import GenerationSettings, TokenizerFingerprint
from kawaneen.generation.prompt import RenderedPrompt, render_generation_prompt
from kawaneen.generation.tokenizer import TokenizerAdapter
from kawaneen.grounding.contracts import ContextPack
from kawaneen.grounding.rendering import render_evidence

CONTEXT_CACHE_SCHEMA_VERSION = 1


def generator_context_fingerprint(
    *,
    query_id: str,
    phase8_selection_sha256: str,
    input_chunk_ids: tuple[str, ...],
    phase9_policy_hash: str,
    tokenizer_fingerprint: TokenizerFingerprint,
    settings: GenerationSettings,
    evidence_token_budget: int,
    fixed_prompt_tokens: int = 0,
    query: str | None = None,
    prompt_template_version: str = "",
    jurisdiction_text: str | None = None,
) -> str:
    """Fingerprint all inputs that can change a generator-specific pack."""

    payload = {
        "query_id": query_id,
        "query_hash": artifact_fingerprint(query) if query is not None else None,
        "phase8_selection_sha256": phase8_selection_sha256,
        "ordered_phase8_input_chunk_ids": list(input_chunk_ids),
        "phase9_policy_hash": phase9_policy_hash,
        "tokenizer_id": tokenizer_fingerprint.identity,
        "tokenizer_revision": tokenizer_fingerprint.revision,
        "prompt_input_token_cap": settings.total_input_tokens,
        "output_reservation": settings.output_reservation,
        "safety_margin": settings.safety_margin,
        "fixed_prompt_tokens": fixed_prompt_tokens,
        "prompt_template_version": prompt_template_version,
        "jurisdiction_text": jurisdiction_text,
        "evidence_token_budget": evidence_token_budget,
        "generation_settings": settings.model_dump(mode="json"),
    }
    return artifact_fingerprint(payload)


def assemble_or_load_generator_context(
    *,
    query: str,
    context_seed: ContextPack,
    tokenizer: TokenizerAdapter,
    assembler_factory: Callable[[int], ContextPack],
    settings: GenerationSettings | None = None,
    phase9_policy_hash: str,
    cache_root: Path,
    jurisdiction_text: str | None = None,
    prompt_renderer: Callable[..., RenderedPrompt] = render_generation_prompt,
    prompt_template_version: str = "",
) -> BudgetedContext:
    """Load a valid private Phase-10 pack or rebuild it from Phase-8/9 inputs."""

    effective_settings = settings or GenerationSettings()
    fixed_prompt_tokens, evidence_budget = calculate_evidence_budget(
        query=query,
        context_pack=context_seed,
        tokenizer=tokenizer,
        settings=effective_settings,
        jurisdiction_text=jurisdiction_text,
        prompt_renderer=prompt_renderer,
    )
    fingerprint = generator_context_fingerprint(
        query_id=context_seed.query_id,
        query=query,
        phase8_selection_sha256=context_seed.phase8_selection_sha256,
        input_chunk_ids=context_seed.input_chunk_ids,
        phase9_policy_hash=phase9_policy_hash,
        tokenizer_fingerprint=tokenizer.fingerprint,
        settings=effective_settings,
        evidence_token_budget=evidence_budget,
        fixed_prompt_tokens=fixed_prompt_tokens,
        prompt_template_version=prompt_template_version,
        jurisdiction_text=jurisdiction_text,
    )
    cache_path = _cache_path(cache_root, context_seed.query_id)
    cached = _load_cached_context(
        cache_path,
        fingerprint=fingerprint,
        query=query,
        tokenizer=tokenizer,
        settings=effective_settings,
        jurisdiction_text=jurisdiction_text,
        prompt_renderer=prompt_renderer,
        prompt_template_version=prompt_template_version,
        expected_budget=evidence_budget,
        expected_fixed_prompt_tokens=fixed_prompt_tokens,
        expected_phase9_hash=phase9_policy_hash,
        expected_seed=context_seed,
    )
    if cached is not None:
        return cached

    budgeted = budget_context(
        query=query,
        context_pack=context_seed,
        tokenizer=tokenizer,
        assembler_factory=assembler_factory,
        settings=effective_settings,
        jurisdiction_text=jurisdiction_text,
        prompt_renderer=prompt_renderer,
    )
    if budgeted.evidence_budget_tokens != evidence_budget:
        raise AssertionError("generator context evidence budget changed during assembly")
    if budgeted.context_pack.token_counter_identity != _tokenizer_identity(tokenizer):
        raise ValueError("generator context was assembled with the wrong tokenizer")
    _write_cached_context(
        cache_path,
        {
            "schema_version": CONTEXT_CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "query_id": context_seed.query_id,
            "phase8_selection_sha256": context_seed.phase8_selection_sha256,
            "ordered_phase8_input_chunk_ids": list(context_seed.input_chunk_ids),
            "phase9_policy_hash": phase9_policy_hash,
            "tokenizer_id": tokenizer.fingerprint.identity,
            "tokenizer_revision": tokenizer.fingerprint.revision,
            "prompt_input_token_cap": effective_settings.total_input_tokens,
            "output_reservation": effective_settings.output_reservation,
            "safety_margin": effective_settings.safety_margin,
            "fixed_prompt_tokens": fixed_prompt_tokens,
            "prompt_template_version": prompt_template_version,
            "evidence_token_budget": evidence_budget,
            "context_pack": budgeted.context_pack.model_dump(mode="json"),
        },
    )
    return budgeted


def _load_cached_context(
    path: Path,
    *,
    fingerprint: str,
    query: str,
    tokenizer: TokenizerAdapter,
    settings: GenerationSettings,
    jurisdiction_text: str | None,
    expected_budget: int,
    expected_fixed_prompt_tokens: int,
    expected_phase9_hash: str,
    expected_seed: ContextPack,
    prompt_renderer: Callable[..., RenderedPrompt],
    prompt_template_version: str,
) -> BudgetedContext | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        value = cast(dict[str, object], raw)
        if value.get("fingerprint") != fingerprint:
            return None
        chunk_ids_value = value.get("ordered_phase8_input_chunk_ids", ())
        if not isinstance(chunk_ids_value, list):
            return None
        chunk_ids = cast(list[object], chunk_ids_value)
        if (
            value.get("schema_version") != CONTEXT_CACHE_SCHEMA_VERSION
            or value.get("query_id") != expected_seed.query_id
            or value.get("phase8_selection_sha256") != expected_seed.phase8_selection_sha256
            or tuple(str(item) for item in chunk_ids) != expected_seed.input_chunk_ids
            or value.get("phase9_policy_hash") != expected_phase9_hash
            or value.get("tokenizer_id") != tokenizer.fingerprint.identity
            or value.get("tokenizer_revision") != tokenizer.fingerprint.revision
            or value.get("prompt_input_token_cap") != settings.total_input_tokens
            or value.get("output_reservation") != settings.output_reservation
            or value.get("safety_margin") != settings.safety_margin
            or value.get("fixed_prompt_tokens") != expected_fixed_prompt_tokens
            or value.get("prompt_template_version") != prompt_template_version
            or value.get("evidence_token_budget") != expected_budget
        ):
            return None
        pack_value = value.get("context_pack")
        if not isinstance(pack_value, dict):
            return None
        pack = ContextPack.model_validate(pack_value)
        if (
            pack.query_id != expected_seed.query_id
            or pack.phase8_selection_sha256 != expected_seed.phase8_selection_sha256
            or pack.input_chunk_ids != expected_seed.input_chunk_ids
            or pack.max_context_tokens != expected_budget
            or pack.token_counter_identity != _tokenizer_identity(tokenizer)
            or pack.token_count > pack.max_context_tokens
        ):
            return None
        rendered = prompt_renderer(
            query,
            pack,
            settings=settings,
            jurisdiction_text=jurisdiction_text,
        )
        prompt_tokens = tokenizer.count(rendered.text)
        if prompt_tokens > settings.total_input_tokens:
            return None
        evidence_tokens = tokenizer.count(render_evidence(pack))
        return BudgetedContext(
            context_pack=pack,
            tokenizer_identity=_tokenizer_identity(tokenizer),
            non_evidence_prompt_tokens=expected_fixed_prompt_tokens,
            evidence_token_count=evidence_tokens,
            prompt_token_count=prompt_tokens,
            evidence_budget_tokens=expected_budget,
            omitted_unit_ids=tuple(item.unit_id for item in pack.omissions),
            gold_evidence_retention=None,
            complete_gold_evidence_retention=None,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cached_context(path: Path, payload: Mapping[str, object]) -> None:
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


def _cache_path(root: Path, query_id: str) -> Path:
    if not query_id or Path(query_id).name != query_id or "/" in query_id:
        raise ValueError("unsafe generator context query ID")
    return root / f"{query_id}.json"


def _tokenizer_identity(tokenizer: TokenizerAdapter) -> str:
    fingerprint = tokenizer.fingerprint
    return ":".join(item for item in (fingerprint.identity, fingerprint.revision) if item)
