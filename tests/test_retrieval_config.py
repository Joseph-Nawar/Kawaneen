from pathlib import Path

import pytest

from kawaneen.retrieval.config import load_phase7_config


def test_phase7_config_contains_frozen_experiment_contract() -> None:
    config = load_phase7_config(Path("configs/retrieval/phase7_baselines.toml"))

    assert config.bootstrap_seed == 20260815
    assert config.bootstrap_replicates == 2000
    assert config.bm25_k1 == 1.2
    assert config.bm25_b == 0.75
    assert config.e5_max_length == 512
    assert config.bge_max_length == 1536
    assert config.model_ids == (
        "intfloat/multilingual-e5-small",
        "BAAI/bge-m3",
    )


def test_phase7_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_phase7_config(tmp_path / "missing.toml")
