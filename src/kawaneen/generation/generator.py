"""Generator protocol shared by deterministic and local adapters."""

from __future__ import annotations

from typing import Protocol

from kawaneen.generation.contracts import GenerationRequest, GenerationResult


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
