from __future__ import annotations

import kawaneen.ui.state as state_module
from kawaneen.ui.client import UiApiError
from kawaneen.ui.config import ModeResolution, UiMode, UiSettings
from kawaneen.ui.demo import DemoClient
from kawaneen.ui.state import UiSessionState, activate_demo_mode, get_context


class _StreamlitState:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}


def test_session_state_records_live_latency_and_ignores_demo_latency() -> None:
    live = UiSessionState(
        settings=UiSettings(mode=UiMode.LIVE),
        resolution=ModeResolution(UiMode.LIVE, "Live"),
    )
    live.record_latency("search", -4)
    live.record_latency("answer", 12.5)
    live.record_latency("unknown", 20)
    assert live.search_latency_ms == [0.0]
    assert live.answer_latency_ms == [12.5]

    demo = UiSessionState(
        settings=UiSettings(mode=UiMode.DEMO),
        resolution=ModeResolution(UiMode.DEMO, "Demo data"),
        demo_activated=True,
    )
    demo.record_latency("search", 99)
    assert demo.active_mode is UiMode.DEMO
    assert demo.status_label == "Demo data"
    assert demo.search_latency_ms == []


def test_session_latency_groups_are_explicitly_endpoint_specific() -> None:
    live = UiSessionState(
        settings=UiSettings(mode=UiMode.LIVE),
        resolution=ModeResolution(UiMode.LIVE, "Live"),
    )

    live.record_latency("search", 10)
    live.record_latency("answer", 20)
    live.record_latency("extract", 30)

    assert live.search_latency_ms == [10]
    assert live.answer_latency_ms == [20]
    assert live.extract_latency_ms == [30]


def test_demo_context_is_cached_and_can_be_activated(monkeypatch) -> None:
    fake = _StreamlitState()
    monkeypatch.setattr(state_module, "st", fake)
    monkeypatch.setattr(
        state_module.UiSettings,
        "from_env",
        classmethod(lambda cls, env=None: UiSettings(mode=UiMode.DEMO)),
    )

    client, session = get_context()
    cached_client, cached_session = get_context()
    assert isinstance(client, DemoClient)
    assert cached_client is client
    assert cached_session is session

    activate_demo_mode()
    activated_client, activated_session = fake.session_state["kawaneen_ui_context"]
    assert isinstance(activated_client, DemoClient)
    assert activated_session is session
    assert activated_session.demo_activated is True


def test_activate_demo_mode_without_context_is_safe(monkeypatch) -> None:
    fake = _StreamlitState()
    monkeypatch.setattr(state_module, "st", fake)

    activate_demo_mode()
    assert fake.session_state == {}


def test_auto_context_keeps_live_client_when_health_is_unavailable(monkeypatch) -> None:
    fake = _StreamlitState()
    monkeypatch.setattr(state_module, "st", fake)
    monkeypatch.setattr(
        state_module.UiSettings,
        "from_env",
        classmethod(lambda cls, env=None: UiSettings(mode=UiMode.AUTO)),
    )

    class BrokenClient:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def health(self):
            raise UiApiError("API_UNAVAILABLE", "offline")

    monkeypatch.setattr(state_module, "HttpUiClient", BrokenClient)

    client, session = get_context()
    assert isinstance(client, BrokenClient)
    assert session.active_mode is None
    assert session.resolution.requires_demo_activation is True
