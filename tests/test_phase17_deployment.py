from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_full_compose_has_required_services_and_loopback_ports() -> None:
    document = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    services = document["services"]
    assert set(services) == {
        "kawaneen-api",
        "kawaneen-ui",
        "qdrant",
        "mlflow",
        "ollama",
        "qdrant-init",
        "ollama-init",
        "hf-model-init",
    }
    assert "127.0.0.1:8000:8000" in services["kawaneen-api"]["ports"]
    assert "127.0.0.1:8501:8501" in services["kawaneen-ui"]["ports"]
    assert "127.0.0.1:6333:6333" in services["qdrant"]["ports"]
    assert "127.0.0.1:5000:5000" in services["mlflow"]["ports"]
    assert "127.0.0.1:11434:11434" in services["ollama"]["ports"]


def test_full_compose_uses_health_ordering_and_read_only_external_artifacts() -> None:
    document = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    services = document["services"]
    api = services["kawaneen-api"]
    assert api["environment"]["KAWANEEN_DENSE_INDEX_BACKEND"] == "qdrant"
    assert api["environment"]["KAWANEEN_OLLAMA_URL"] == "http://ollama:11434"
    assert api["environment"]["KAWANEEN_QDRANT_URL"] == "http://qdrant:6333"
    assert any("/app/artifacts:ro" in mount for mount in api["volumes"])
    assert services["qdrant-init"]["depends_on"]["qdrant"]["condition"] == "service_healthy"
    assert services["ollama-init"]["depends_on"]["ollama"]["condition"] == "service_healthy"
    assert services["hf-model-init"]["environment"]["HF_HOME"] == "/opt/huggingface"
    assert "huggingface_cache:/opt/huggingface" in services["hf-model-init"]["volumes"]
    assert services["kawaneen-api"]["depends_on"]["hf-model-init"]["condition"] == (
        "service_completed_successfully"
    )
    assert services["kawaneen-api"]["depends_on"]["qdrant-init"]["condition"] == (
        "service_completed_successfully"
    )


def test_full_runtime_context_does_not_copy_private_artifacts() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "artifacts/private" in dockerignore
    assert ".env" in dockerignore
    assert "data/raw" in dockerignore
