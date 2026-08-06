from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kawaneen.core.config import Settings


def test_settings_defaults(monkeypatch) -> None:
    names = (
        "KAWANEEN_ENVIRONMENT",
        "KAWANEEN_LOG_LEVEL",
        "KAWANEEN_LOG_FORMAT",
        "KAWANEEN_DATA_DIRECTORY",
        "KAWANEEN_ARTIFACTS_DIRECTORY",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.log_format == "console"
    assert settings.data_directory == Path("data")
    assert settings.artifacts_directory.name == "artifacts"


def test_settings_environment_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KAWANEEN_ENVIRONMENT", "test")
    monkeypatch.setenv("KAWANEEN_LOG_LEVEL", "debug")
    monkeypatch.setenv("KAWANEEN_LOG_FORMAT", "json")
    monkeypatch.setenv("KAWANEEN_DATA_DIRECTORY", str(tmp_path / "raw-data"))
    monkeypatch.setenv("KAWANEEN_ARTIFACTS_DIRECTORY", str(tmp_path / "artifacts"))

    settings = Settings(_env_file=None)
    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "json"
    assert settings.data_directory == tmp_path / "raw-data"
    assert settings.artifacts_directory == tmp_path / "artifacts"
    assert not (tmp_path / "raw-data").exists()
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("KAWANEEN_ENVIRONMENT", "invalid"),
        ("KAWANEEN_LOG_LEVEL", "verbose"),
        ("KAWANEEN_LOG_FORMAT", "xml"),
    ],
)
def test_invalid_settings_fail_clearly(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=name.removeprefix("KAWANEEN_").lower()):
        Settings(_env_file=None)
