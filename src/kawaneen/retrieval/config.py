"""Explicit Phase 7 configuration loading with no import-time I/O."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Phase7Config:
    dataset_version: str
    chunk_policy_id: str
    normalization_policy_ids: tuple[str, ...]
    bm25_k1: float
    bm25_b: float
    bootstrap_seed: int
    bootstrap_replicates: int
    bootstrap_confidence: float
    e5_max_length: int
    bge_max_length: int
    model_ids: tuple[str, ...]
    e5_batch_size: int
    bge_batch_size: int
    dense_device: str
    private_root: Path
    tracked_manifest_root: Path
    tracked_evaluation_root: Path


def load_phase7_config(
    path: Path = Path("configs/retrieval/phase7_baselines.toml"),
) -> Phase7Config:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    experiment = raw["experiment"]
    dense = raw["dense"]
    paths = raw["paths"]
    models = tuple(str(model) for model in dense["model_ids"])
    if models != ("intfloat/multilingual-e5-small", "BAAI/bge-m3"):
        raise ValueError("Phase 7 requires the fixed E5-small and BGE-M3 model set")
    return Phase7Config(
        dataset_version=str(experiment["dataset_version"]),
        chunk_policy_id=str(experiment["chunk_policy_id"]),
        normalization_policy_ids=tuple(
            str(item) for item in experiment["normalization_policy_ids"]
        ),
        bm25_k1=float(experiment["bm25_k1"]),
        bm25_b=float(experiment["bm25_b"]),
        bootstrap_seed=int(experiment["bootstrap_seed"]),
        bootstrap_replicates=int(experiment["bootstrap_replicates"]),
        bootstrap_confidence=float(experiment["bootstrap_confidence"]),
        e5_max_length=int(dense["e5_max_length"]),
        bge_max_length=int(dense["bge_max_length"]),
        model_ids=models,
        e5_batch_size=int(dense["e5_batch_size"]),
        bge_batch_size=int(dense["bge_batch_size"]),
        dense_device=str(dense.get("device", "cpu")),
        private_root=Path(str(paths["private_root"])),
        tracked_manifest_root=Path(str(paths["tracked_manifest_root"])),
        tracked_evaluation_root=Path(str(paths["tracked_evaluation_root"])),
    )
