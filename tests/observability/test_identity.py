from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from kawaneen.observability.identity import (
    ServingIdentity,
    load_serving_identity,
    verify_tracked_serving_identity,
)

ROOT = Path(__file__).parents[2]


def test_authoritative_identity_is_stable_and_committed_identity_matches() -> None:
    first = ServingIdentity.build(ROOT / "data")
    second = ServingIdentity.build(ROOT / "data")

    assert first == second
    assert first.configuration_version == second.configuration_version
    assert (
        verify_tracked_serving_identity(
            ROOT / "data", ROOT / "data/manifests/observability/phase16_serving_identity.json"
        )
        == first
    )


def test_mapping_order_does_not_change_configuration_version() -> None:
    identity = ServingIdentity.build(ROOT / "data")
    reordered = ServingIdentity.from_mapping(
        json.loads(json.dumps(identity.to_dict(), ensure_ascii=False))
    )

    assert reordered.configuration_version == identity.configuration_version
    assert identity.canonical_bytes() == reordered.canonical_bytes()


@pytest.mark.parametrize(
    "field",
    (
        "corpus_version",
        "embedding_revision",
        "reranker_revision",
        "prompt_version_hash",
    ),
)
def test_authoritative_identity_changes_when_frozen_inputs_change(field: str) -> None:
    identity = ServingIdentity.build(ROOT / "data")
    if field == "corpus_version":
        changed = replace(identity, corpus_version="0" * 64)
    elif field == "embedding_revision":
        changed = replace(
            identity,
            embedding=replace(identity.embedding, revision="changed-embedding-revision"),
        )
    elif field == "reranker_revision":
        changed = replace(
            identity,
            reranker=replace(identity.reranker, revision="changed-reranker-revision"),
        )
    else:
        changed = replace(
            identity,
            prompt=replace(identity.prompt, version_hash="changed-prompt-hash"),
        )

    assert changed.configuration_version != identity.configuration_version


def test_identity_loader_rejects_a_modified_configuration_version(tmp_path: Path) -> None:
    identity = ServingIdentity.build(ROOT / "data")
    path = tmp_path / "identity.json"
    path.write_bytes(identity.canonical_bytes())
    value = json.loads(path.read_text(encoding="utf-8"))
    value["configuration_version"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration_version"):
        load_serving_identity(path)


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ({"schema_version": "wrong"}, "schema_version"),
        ({"schema_version": "phase16-serving-identity-v1"}, "configuration_version"),
    ),
)
def test_identity_loader_rejects_invalid_top_level_fields(
    tmp_path: Path, value: dict[str, object], message: str
) -> None:
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_serving_identity(path)


def test_identity_loader_reports_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        load_serving_identity(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        load_serving_identity(invalid)


def test_identity_loader_rejects_invalid_nested_fields(tmp_path: Path) -> None:
    value = ServingIdentity.build(ROOT / "data").to_dict()
    value["embedding"] = {"model_id": "only-one-field"}
    value.pop("configuration_version")
    value["configuration_version"] = hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="embedding has invalid fields"):
        load_serving_identity(path)


def test_identity_build_reports_missing_authoritative_sources(tmp_path: Path) -> None:
    from kawaneen.api.runtime import ExpectedAssetUnavailable

    with pytest.raises(ExpectedAssetUnavailable, match="frozen retrieval configuration"):
        ServingIdentity.build(tmp_path)


def test_identity_build_uses_tracked_sources_only(tmp_path: Path) -> None:
    identity = ServingIdentity.build(ROOT / "data")

    assert identity.corpus_version
    assert not any("private" in str(path) for path in identity.source_paths)
    assert not tmp_path.joinpath("private").exists()
