"""Generation prompt budgeting over complete Phase-9 evidence units."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kawaneen.generation.contracts import GenerationSettings
from kawaneen.generation.prompt import RenderedPrompt, render_generation_prompt
from kawaneen.generation.tokenizer import TokenizerAdapter
from kawaneen.grounding.contracts import ContextPack
from kawaneen.grounding.rendering import render_evidence


@dataclass(frozen=True, slots=True)
class BudgetedContext:
    context_pack: ContextPack
    tokenizer_identity: str
    non_evidence_prompt_tokens: int
    evidence_token_count: int
    prompt_token_count: int
    evidence_budget_tokens: int
    omitted_unit_ids: tuple[str, ...]
    gold_evidence_retention: float | None
    complete_gold_evidence_retention: float | None


def calculate_evidence_budget(
    *,
    query: str,
    context_pack: ContextPack,
    tokenizer: TokenizerAdapter,
    settings: GenerationSettings,
    jurisdiction_text: str | None = None,
    prompt_renderer: Callable[..., RenderedPrompt] = render_generation_prompt,
) -> tuple[int, int]:
    """Return fixed prompt overhead and remaining evidence-token budget."""

    empty_pack = context_pack.model_copy(update={"evidence": (), "token_count": 0})
    non_evidence_prompt_tokens = tokenizer.count(
        prompt_renderer(
            query,
            empty_pack,
            settings=settings,
            jurisdiction_text=jurisdiction_text,
        ).text
    )
    evidence_budget = (
        settings.total_input_tokens
        - non_evidence_prompt_tokens
        - settings.safety_margin
    )
    if evidence_budget <= 0:
        raise ValueError("non-evidence prompt overhead leaves no evidence budget")
    return non_evidence_prompt_tokens, evidence_budget


def budget_context(
    *,
    query: str,
    context_pack: ContextPack,
    tokenizer: TokenizerAdapter,
    assembler_factory: Callable[[int], ContextPack] | None = None,
    settings: GenerationSettings | None = None,
    gold_unit_ids: Sequence[str] = (),
    jurisdiction_text: str | None = None,
    prompt_renderer: Callable[..., RenderedPrompt] = render_generation_prompt,
) -> BudgetedContext:
    effective_settings = settings or GenerationSettings()
    non_evidence_prompt_tokens, evidence_budget = calculate_evidence_budget(
        query=query,
        context_pack=context_pack,
        tokenizer=tokenizer,
        settings=effective_settings,
        jurisdiction_text=jurisdiction_text,
        prompt_renderer=prompt_renderer,
    )
    selected_pack = assembler_factory(evidence_budget) if assembler_factory else context_pack
    rendered = prompt_renderer(
        query,
        selected_pack,
        settings=effective_settings,
        jurisdiction_text=jurisdiction_text,
    )
    prompt_token_count = tokenizer.count(rendered.text)
    if prompt_token_count > effective_settings.total_input_tokens:
        raise ValueError("rendered generation prompt exceeds input token budget")
    evidence_token_count = tokenizer.count(render_evidence(selected_pack))
    gold = frozenset(gold_unit_ids)
    retained = frozenset(item.unit_id for item in selected_pack.evidence)
    if gold:
        retention = len(gold & retained) / len(gold)
        complete = 1.0 if gold <= retained else 0.0
    else:
        retention = None
        complete = None
    return BudgetedContext(
        context_pack=selected_pack,
        tokenizer_identity=_tokenizer_identity(tokenizer),
        non_evidence_prompt_tokens=non_evidence_prompt_tokens,
        evidence_token_count=evidence_token_count,
        prompt_token_count=prompt_token_count,
        evidence_budget_tokens=evidence_budget,
        omitted_unit_ids=tuple(item.unit_id for item in selected_pack.omissions),
        gold_evidence_retention=retention,
        complete_gold_evidence_retention=complete,
    )


def _tokenizer_identity(tokenizer: TokenizerAdapter) -> str:
    fingerprint = tokenizer.fingerprint
    return ":".join(item for item in (fingerprint.identity, fingerprint.revision) if item)
