from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kawaneen.api.composition import load_frozen_serving_configuration

if TYPE_CHECKING:
    from kawaneen.api.composition import FrozenServingConfiguration


_MODEL_METADATA_FILES = ("config.json", "config.yaml", "tokenizer.json", "tokenizer_config.json")


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    model_id: str
    revision: str
    path: Path


def _frozen_models(configuration: FrozenServingConfiguration) -> tuple[tuple[str, str], ...]:
    return (
        (configuration.dense_model_id, configuration.dense_model_revision),
        (configuration.reranker.model_id, configuration.reranker.model_revision),
    )


def _validate_snapshot(model_id: str, revision: str, path: Path) -> ModelSnapshot:
    if not path.is_dir() or not any((path / name).is_file() for name in _MODEL_METADATA_FILES):
        raise RuntimeError(f"frozen snapshot metadata is missing for {model_id} {revision}")
    return ModelSnapshot(model_id=model_id, revision=revision, path=path)


def bootstrap_models(
    data_directory: Path,
    cache_directory: Path,
    snapshot_downloader: Callable[..., str | Path],
) -> tuple[ModelSnapshot, ...]:
    configuration = load_frozen_serving_configuration(data_directory)
    hub_cache = cache_directory / "hub"
    snapshots: list[ModelSnapshot] = []
    for model_id, revision in _frozen_models(configuration):
        snapshot_path = Path(
            snapshot_downloader(
                repo_id=model_id,
                revision=revision,
                cache_dir=hub_cache,
                local_files_only=False,
            )
        )
        snapshots.append(_validate_snapshot(model_id, revision, snapshot_path))
    return tuple(snapshots)


def main() -> None:
    from huggingface_hub import snapshot_download

    data_directory = Path(os.environ.get("KAWANEEN_DATA_DIRECTORY", "/app/data"))
    cache_directory = Path(os.environ.get("HF_HOME", "/opt/huggingface"))
    for snapshot in bootstrap_models(data_directory, cache_directory, snapshot_download):
        print(f"HF snapshot ready: {snapshot.model_id} {snapshot.revision}")


if __name__ == "__main__":
    main()
