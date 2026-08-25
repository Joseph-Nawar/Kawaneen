"""Small session-state helpers kept separate from Streamlit page rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

from kawaneen.ui.config import ModeResolution, UiMode, UiSettings


@dataclass
class UiSessionState:
    settings: UiSettings
    resolution: ModeResolution
    demo_activated: bool = False
    search_latency_ms: list[float] = field(default_factory=list)
    answer_latency_ms: list[float] = field(default_factory=list)
    extract_latency_ms: list[float] = field(default_factory=list)

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
