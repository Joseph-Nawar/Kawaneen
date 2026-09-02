"""Lazy local-only text model adapter used by Phase 15 diagnostics."""

# pyright: basic

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any


class LocalInstructionModel:
    """Deterministic Transformers adapter; never downloads during scoring."""

    def __init__(
        self, model_id: str, revision: str, *, dtype: str = "bfloat16", device: str = "cpu"
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.dtype = dtype
        self.device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self.snapshot: Path | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        snapshot = Path(
            snapshot_download(
                self.model_id,
                revision=self.revision,
                local_files_only=True,
            )
        )
        self.snapshot = snapshot
        self._tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self._model.to(self.device)
        self._model.eval()

    def generate(self, instruction: str, *, max_new_tokens: int = 160) -> str:
        self.load()
        assert self._model is not None and self._tokenizer is not None
        import torch

        messages = [{"role": "user", "content": instruction}]
        encoded = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        encoded = encoded.to(self.device)
        with torch.inference_mode():
            output = self._model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, encoded.shape[-1] :]
        return str(self._tokenizer.decode(generated, skip_special_tokens=True)).strip()


def parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class LocalOllamaInstructionModel:
    """Fixed local Ollama model adapter for diagnostic assistance."""

    def __init__(self, model_tag: str, *, seed: int = 20260826) -> None:
        self.model_tag = model_tag
        self.seed = seed

    def generate(self, instruction: str, *, max_new_tokens: int = 160) -> str:
        payload = json.dumps(
            {
                "model": self.model_tag,
                "prompt": instruction,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "seed": self.seed,
                    "num_predict": max_new_tokens,
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict) or not isinstance(decoded.get("response"), str):
            raise RuntimeError("local Ollama response did not contain text")
        return str(decoded["response"]).strip()
