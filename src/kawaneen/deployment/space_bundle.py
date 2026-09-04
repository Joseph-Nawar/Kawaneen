from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def export_bundle(output: Path = ROOT / "build/phase17-space") -> Path:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _copy("pyproject.toml", output / "pyproject.toml")
    _copy("uv.lock", output / "uv.lock")
    _copy("README.md", output / "README.md")
    _copy("LICENSE", output / "LICENSE")
    dockerfile = (ROOT / "deploy/hf-space/Dockerfile").read_text(encoding="utf-8")
    dockerfile = dockerfile.replace(
        "COPY deploy/hf-space/entrypoint.sh /app/entrypoint.sh",
        "COPY entrypoint.sh /app/entrypoint.sh",
    )
    (output / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    _copy("deploy/hf-space/entrypoint.sh", output / "entrypoint.sh")
    _copy("deploy/hf-space/README.template.md", output / "README.template.md")
    shutil.copytree(ROOT / "src", output / "src")
    shutil.copytree(ROOT / "data/demo", output / "data/demo")
    files = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    )
    manifest = {
        "schema": "phase17-space-bundle-v1",
        "profile": "public-demo",
        "publication_status": "NOT_PUBLISHED_USER_APPROVAL_REQUIRED",
        "files": files,
        "sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files
        },
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def _copy(relative: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, destination)


__all__ = ["ROOT", "export_bundle"]
