"""Generic lazy local Transformers adapter; no weights load at import time."""

from __future__ import annotations

from typing import Any, Protocol, cast

from kawaneen.generation.abstention import invalid_generation_result
from kawaneen.generation.contracts import (
    GenerationRequest,
    GenerationResult,
    GenerationSettings,
    ModelCandidate,
    parse_model_output,
)
from kawaneen.generation.prompt import render_generation_prompt


class TransformersRuntime(Protocol):
    def generate(self, prompt: str, settings: GenerationSettings) -> str: ...


class TransformersLoader(Protocol):
    def load(self, candidate: ModelCandidate) -> TransformersRuntime: ...


class TransformersGenerator:
    def __init__(
        self,
        *,
        candidate: ModelCandidate,
        loader: TransformersLoader | None = None,
    ) -> None:
        self.candidate = candidate
        self.loader = loader or LocalTransformersLoader()
        self._runtime: TransformersRuntime | None = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.candidate.hf_revision is None:
            return invalid_generation_result("Transformers execution requires a locked HF revision")
        try:
            if self._runtime is None:
                self._runtime = self.loader.load(self.candidate)
            prompt = render_generation_prompt(
                request.query,
                request.context_pack,
                settings=request.settings,
                jurisdiction_text=request.jurisdiction_text,
            )
            raw = self._runtime.generate(prompt.text, request.settings)
            output = parse_model_output(raw)
            return GenerationResult(decision=output.decision, claims=output.claims)
        except Exception as error:  # fail closed; adapters do not retry
            return invalid_generation_result(str(error))


class LocalTransformersLoader:
    def load(self, candidate: ModelCandidate) -> TransformersRuntime:
        if candidate.hf_revision is None:
            raise ValueError("HF revision must be locked before loading")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = cast(
            Any,
            AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                candidate.hf_identity,
                revision=candidate.hf_revision,
                local_files_only=True,
            ),
        )
        model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                candidate.hf_identity,
                revision=candidate.hf_revision,
                local_files_only=True,
            ),
        )
        return _TransformersRuntime(tokenizer, model)


class _TransformersRuntime:
    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, prompt: str, settings: GenerationSettings) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        generated = self.model.generate(
            **encoded,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.do_sample,
            temperature=settings.temperature if settings.do_sample else None,
        )
        prompt_length = encoded["input_ids"].shape[-1]
        return str(self.tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True))
