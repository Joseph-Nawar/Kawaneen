"""Load strict, version-controlled acquisition specifications."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import ValidationError

from kawaneen.acquisition.models import SourceSpecification


def load_specification(path: Path) -> SourceSpecification:
    """Load one TOML acquisition specification without network or writes."""

    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
        return SourceSpecification.model_validate(payload)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid acquisition specification {path}: {exc}") from exc


def load_specifications(
    directory: Path = Path("data/manifests/acquisition_specs"),
) -> dict[str, SourceSpecification]:
    """Load all deterministic TOML specifications sorted by filename."""

    if not directory.is_dir():
        raise ValueError(f"acquisition specification directory does not exist: {directory}")
    specifications: dict[str, SourceSpecification] = {}
    for path in sorted(directory.glob("*.toml")):
        specification = load_specification(path)
        if specification.source_id in specifications:
            raise ValueError(f"duplicate acquisition specification: {specification.source_id}")
        specifications[specification.source_id] = specification
    return specifications
