from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kawaneen.api.composition import load_frozen_serving_configuration
from kawaneen.normalization import get_policy
from regression.conftest import CASES_PATH, LOCK_PATH, ROOT, load_lock

pytestmark = pytest.mark.regression


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_regression_lock_matches_every_authoritative_identity_and_hash() -> None:
    lock = load_lock()
    normalization = lock["normalization"]
    parsing = lock["parsing"]
    chunking = lock["chunking"]
    fixtures = lock["fixtures"]

    assert normalization["policy_hash"] == get_policy("arabic-light-v1").policy_hash
    assert _sha256(ROOT / normalization["source_path"]) == normalization["source_sha256"]
    assert _sha256(ROOT / parsing["config_path"]) == parsing["config_sha256"]
    assert _sha256(ROOT / parsing["source_path"]) == parsing["source_sha256"]
    assert _sha256(ROOT / parsing["adapter_path"]) == parsing["adapter_sha256"]
    assert _sha256(ROOT / chunking["config_path"]) == chunking["config_sha256"]
    assert _sha256(ROOT / chunking["source_path"]) == chunking["source_sha256"]
    assert _sha256(ROOT / fixtures["pdf_path"]) == fixtures["pdf_sha256"]
    assert _sha256(CASES_PATH) == fixtures["cases_sha256"]
    generation = lock["generation"]
    generation_path = ROOT / generation["configuration_path"]
    generation_config = json.loads(generation_path.read_text(encoding="utf-8"))
    assert _sha256(generation_path) == generation["configuration_sha256"]
    assert generation_config["template_version"] == generation["prompt_identity"]
    assert generation_config["system_rules_version"] == generation["system_rules_identity"]
    answerability = lock["answerability"]
    assert _sha256(ROOT / answerability["source_path"]) == answerability["source_sha256"]


def test_regression_lock_matches_authoritative_serving_configuration() -> None:
    lock = load_lock()
    configuration = load_frozen_serving_configuration(ROOT / "data")

    assert lock["dense_model"]["model_id"] == configuration.dense_model_id
    assert lock["dense_model"]["revision"] == configuration.dense_model_revision
    assert lock["reranker"]["model_id"] == configuration.reranker.model_id
    assert lock["reranker"]["revision"] == configuration.reranker.model_revision
    assert lock["fusion"] == {
        **lock["fusion"],
        "sparse_top_k": configuration.fusion.sparse_top_k,
        "dense_top_k": configuration.fusion.dense_top_k,
        "sparse_weight": configuration.fusion.sparse_weight,
        "dense_weight": configuration.fusion.dense_weight,
        "rrf_k": configuration.fusion.rrf_k,
        "candidate_k": configuration.fusion.candidate_k,
        "serving_depth": configuration.reranker.serving_depth,
        "scoring_contract": configuration.reranker.scoring_contract,
    }
    assert _sha256(ROOT / lock["fusion"]["config_path"]) == lock["fusion"]["config_sha256"]
    assert _sha256(ROOT / lock["reranker"]["lock_path"]) == lock["reranker"]["lock_sha256"]


def test_regression_cases_and_lock_are_not_auto_rewritten() -> None:
    assert CASES_PATH.name == "phase14_cases.json"
    assert LOCK_PATH.name == "phase14_regression_lock.json"
    assert json.loads(CASES_PATH.read_text(encoding="utf-8"))["schema_version"] == 2
    assert load_lock()["status"] == "phase14_public_synthetic_regression_lock"
