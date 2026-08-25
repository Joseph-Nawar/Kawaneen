"""Configuration and explicit live/demo mode resolution for the UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from os import environ


class UiMode(StrEnum):
    AUTO = "auto"
    LIVE = "live"
    DEMO = "demo"


@dataclass(frozen=True)
class UiSettings:
    mode: UiMode = UiMode.AUTO
    api_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 65.0

    @classmethod
    def from_env(cls, values: Mapping[str, str] | None = None) -> UiSettings:
        source = environ if values is None else values
        try:
            mode = UiMode(source.get("KAWANEEN_UI_MODE", UiMode.AUTO))
        except ValueError:
            mode = UiMode.AUTO
        api_url = source.get("KAWANEEN_API_URL", cls.api_url).rstrip("/") or cls.api_url
        try:
            timeout = max(1.0, min(float(source.get("KAWANEEN_UI_TIMEOUT", "65")), 120.0))
        except ValueError:
            timeout = cls.timeout_seconds
        return cls(mode=mode, api_url=api_url, timeout_seconds=timeout)


@dataclass(frozen=True)
class HealthProbe:
    available: bool
    ready: bool
    detail: str = ""

    @classmethod
    def unavailable(cls, detail: str) -> HealthProbe:
        return cls(available=False, ready=False, detail=detail)

    @classmethod
    def ready_probe(cls) -> HealthProbe:
        return cls(available=True, ready=True)


@dataclass(frozen=True)
class ModeResolution:
    active_mode: UiMode | None
    status_label: str
    detail: str = ""
    requires_demo_activation: bool = False


def resolve_mode(settings: UiSettings, health: HealthProbe | None) -> ModeResolution:
    if settings.mode is UiMode.DEMO:
        return ModeResolution(UiMode.DEMO, "Demo data", "Synthetic fixtures are active.")
    if health is not None and health.available and health.ready:
        return ModeResolution(UiMode.LIVE, "Live API", "Connected to the Phase 12 API.")
    detail = health.detail if health is not None else "Live API health has not been checked."
    if settings.mode is UiMode.AUTO:
        return ModeResolution(None, "Degraded", detail, requires_demo_activation=True)
    return ModeResolution(None, "Degraded", detail)
