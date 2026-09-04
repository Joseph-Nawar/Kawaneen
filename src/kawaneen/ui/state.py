"""Small session-state helpers kept separate from Streamlit page rendering."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import streamlit as st

from kawaneen.core.jurisdiction import Jurisdiction
from kawaneen.ui.client import HttpUiClient, UiApiError, UiClient
from kawaneen.ui.config import HealthProbe, ModeResolution, UiMode, UiSettings, resolve_mode
from kawaneen.ui.demo import DemoClient

_VISUAL_QA_SCENARIOS = {
    "search_arabic",
    "ask_grounded",
    "extract_structured",
}


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
        seed_visual_qa_state(*current)
        return current
    if settings.mode is UiMode.DEMO:
        client: UiClient = DemoClient()
        resolution = ModeResolution(UiMode.DEMO, "Demo data", "Synthetic fixtures are active.")
    else:
        live_client = HttpUiClient(
            settings.api_url,
            timeout=settings.timeout_seconds,
            jurisdiction=(Jurisdiction.KAWANEEN_DEMO if settings.public_demo else Jurisdiction.SA),
        )
        try:
            health = live_client.health()
            probe = HealthProbe(True, health.status == "ready")
        except UiApiError as error:
            probe = HealthProbe.unavailable(error.message)

        client = live_client
        resolution = resolve_mode(settings, HealthProbe(probe.available, probe.ready, probe.detail))
    state = UiSessionState(settings=settings, resolution=resolution)
    st.session_state[context_key] = (client, state)
    seed_visual_qa_state(client, state)
    return client, state


def seed_visual_qa_state(client: UiClient, state: UiSessionState) -> None:
    """Seed a populated synthetic page for bounded screenshot review.

    This is intentionally gated by demo mode and an explicit URL query parameter.
    It never calls the Phase 12 API, reads private data, or changes ordinary demo
    behavior when ``kawaneen_demo_state`` is absent.
    """

    if state.active_mode is not UiMode.DEMO:
        return
    query_params = cast(Mapping[str, object], getattr(st, "query_params", {}))
    query_scenario = str(query_params.get("kawaneen_demo_state", "")).strip()
    scenario = (query_scenario or os.environ.get("KAWANEEN_UI_VISUAL_QA", "")).strip()
    if scenario not in _VISUAL_QA_SCENARIOS:
        return
    if st.session_state.get("_visual_qa_scenario") == scenario:
        return

    synthetic = DemoClient()
    if scenario == "search_arabic":
        query = "ما هي مدة الاعتراض؟"
        st.session_state["search_response"] = synthetic.search(query)
        st.session_state["search_query"] = query
        st.session_state["search_query_value"] = query
    elif scenario == "ask_grounded":
        question = "ما هي مدة الاعتراض؟"
        st.session_state["ask_question"] = question
        st.session_state["answer_response"] = synthetic.answer(question)
    else:
        source_text = (
            "يلتزم الطرف بالسداد خلال ثلاثين يوماً.\n\nيجوز تمديد المدة في الحالات المحددة نظاماً."
        )
        st.session_state["source_text"] = source_text
        st.session_state["extraction_mode_select"] = "Deterministic"
        st.session_state["extraction_results"] = [
            ("segment-001", synthetic.extract(source_text, "deterministic"))
        ]
        st.session_state["extraction_mode"] = "deterministic"
    st.session_state["_visual_qa_scenario"] = scenario


def activate_demo_mode() -> None:
    current = st.session_state.get("kawaneen_ui_context")
    if current is None:
        return
    _client, state = current
    state.demo_activated = True
    st.session_state["kawaneen_ui_context"] = (DemoClient(), state)
