from __future__ import annotations

import json
from pathlib import Path


def test_space_export_contains_allowlisted_runtime_and_hash_manifest(tmp_path: Path) -> None:
    from kawaneen.deployment.space_bundle import export_bundle

    output = export_bundle(tmp_path / "space")
    assert (output / "Dockerfile").is_file()
    assert (output / "entrypoint.sh").is_file()
    assert (output / "README.md").is_file()
    assert (output / "src/kawaneen/demo").is_dir()
    assert (output / "data/demo/vectors.npy").is_file()
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "phase17-space-bundle-v1"
    names = set(manifest["files"])
    assert not any(name.startswith(".git") for name in names)
    assert not any("artifacts/private" in name for name in names)
    assert not any(name.endswith(".env") for name in names)
    assert all((output / name).is_file() for name in names)


def test_space_entrypoint_has_one_public_port_and_no_sidecar_services() -> None:
    from kawaneen.deployment.space_bundle import ROOT

    dockerfile = (ROOT / "deploy/hf-space/Dockerfile").read_text(encoding="utf-8")
    entrypoint = (ROOT / "deploy/hf-space/entrypoint.sh").read_text(encoding="utf-8")
    assert "7860" in dockerfile
    assert "127.0.0.1:8000" in entrypoint
    assert "streamlit" in entrypoint
    assert "HF_HUB_OFFLINE" in dockerfile
    assert "614241f622f53c4eeff9890bdc4f31cfecc418b3" in dockerfile
    assert "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e" in dockerfile
    assert all(name not in entrypoint.lower() for name in ("qdrant", "mlflow", "ollama"))
