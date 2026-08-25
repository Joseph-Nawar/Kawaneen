"""Small session-state helpers kept separate from Streamlit page rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from kawaneen.ui.client import HttpUiClient, UiApiError, UiClient
from kawaneen.ui.config import HealthProbe, ModeResolution, UiMode, UiSettings, resolve_mode
from kawaneen.ui.demo import DemoClient


@dataclass
class UiSessionState:
    settings: UiSettings
    resolution: ModeResolution
    demo_activated: bool = False
    search_latency_ms: list[float] = field(default_factory=lambda: list[float]())
    answer_latency_ms: list[float] = field(default_factory=lambda: list[float]())
    extract_latency_ms: list[float] = field(default_factory=lambda: list[float]())

    @property
    def active_mode(self) -> UiMode | None:
        if self.demo_activated:
            return UiMode.DEMO
        return self.resolution.active_mode

    @property
    def status_label(self) -> str:
        return "Demo data" if self.demo_activated else self.resolution.status_label

    def record_latency(self, kind: str, latency_ms: float) -> None:
        if self.active_mode is UiMode.DEMO:
            return
        values = {
            "search": self.search_latency_ms,
            "answer": self.answer_latency_ms,
            "extract": self.extract_latency_ms,
        }.get(kind)
        if values is not None:
            values.append(max(0.0, latency_ms))


def get_context() -> tuple[UiClient, UiSessionState]:
    settings = UiSettings.from_env()
    context_key = "kawaneen_ui_context"
    current = st.session_state.get(context_key)
    if current is not None and current[1].settings == settings:
        return current
    if settings.mode is UiMode.DEMO:
        client: UiClient = DemoClient()
        resolution = ModeResolution(UiMode.DEMO, "Demo data", "Synthetic fixtures are active.")
    else:
        live_client = HttpUiClient(settings.api_url, timeout=settings.timeout_seconds)
        try:
            health = live_client.health()
            probe = HealthProbe(True, health.status == "ready")
        except UiApiError as error:
            probe = HealthProbe.unavailable(error.message)

        client = live_client
        resolution = resolve_mode(settings, HealthProbe(probe.available, probe.ready, probe.detail))
    state = UiSessionState(settings=settings, resolution=resolution)
    st.session_state[context_key] = (client, state)
    return client, state


def activate_demo_mode() -> None:
    current = st.session_state.get("kawaneen_ui_context")
    if current is None:
        return
    _client, state = current
    state.demo_activated = True
    st.session_state["kawaneen_ui_context"] = (DemoClient(), state)
