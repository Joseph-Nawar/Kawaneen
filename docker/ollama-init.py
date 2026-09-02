from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

from kawaneen.generation.ollama import normalize_sha256_digest


def request(base: str, path: str, payload: dict[str, object] | None = None) -> object:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    with urlopen(
        Request(
            base.rstrip("/") + path, data=body, headers=headers, method="POST" if body else "GET"
        ),
        timeout=10,
    ) as response:
        return json.loads(response.read().decode())


def main() -> None:
    base = os.environ.get("KAWANEEN_OLLAMA_URL", "http://ollama:11434")
    data = Path(os.environ.get("KAWANEEN_DATA_DIRECTORY", "/app/data"))
    lock_path = (
        Path(os.environ.get("KAWANEEN_ARTIFACTS_DIRECTORY", "/app/artifacts"))
        / "private"
        / "phase10_generation"
        / "qwen-ollama-model-lock.json"
    )
    selected = json.loads(
        (data / "manifests/generation/phase10_selected_configuration.json").read_text()
    )
    expected = selected["model"]
    model = expected["ollama_tag"]
    digest = normalize_sha256_digest(expected["ollama_digest"])
    for _ in range(60):
        try:
            request(base, "/api/tags")
            break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError("Ollama API did not become ready")
    tags = request(base, "/api/tags")
    installed = next((item for item in tags.get("models", []) if item.get("name") == model), None)
    if installed is None:
        request(base, "/api/pull", {"name": model, "stream": False})
        tags = request(base, "/api/tags")
        installed = next(
            (item for item in tags.get("models", []) if item.get("name") == model), None
        )
    if installed is None or normalize_sha256_digest(installed.get("digest")) != digest:
        raise RuntimeError("installed Ollama model does not match the frozen Phase 10 digest")
    if not lock_path.is_file():
        raise RuntimeError("authoritative local Ollama lock is missing")
    lock = json.loads(lock_path.read_text())
    if lock.get("model") != model or normalize_sha256_digest(lock.get("digest")) != digest:
        raise RuntimeError("local Ollama lock disagrees with the frozen Phase 10 selection")
    print(f"Ollama model ready: {model} {digest}")


if __name__ == "__main__":
    main()
