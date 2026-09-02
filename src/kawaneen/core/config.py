"""Validated application settings with no filesystem side effects."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["console", "json"]


class Settings(BaseSettings):
    """Runtime settings loaded from defaults and ``KAWANEEN_`` variables."""

    model_config = SettingsConfigDict(env_prefix="KAWANEEN_", extra="ignore")

    environment: Environment = "development"
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "console"
    data_directory: Path = Path("data")
    artifacts_directory: Path = Path("artifacts")
    observability_enabled: bool = False
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_serving_experiment: str = "kawaneen-serving"
    mlflow_repro_experiment: str = "kawaneen-reproducibility"

    @field_validator("environment", "log_format", mode="before")
    @classmethod
    def normalize_lowercase_values(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.upper()
        return value
