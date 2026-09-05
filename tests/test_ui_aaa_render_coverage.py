from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from kawaneen.ui.client import UiClient
from kawaneen.ui.config import ModeResolution, UiMode, UiSettings
from kawaneen.ui.demo import DemoClient
from kawaneen.ui.state import UiSessionState


class FakeStreamlit:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = values or {}
        self.session_state: dict[str, object] = {}

    def __enter__(self) -> FakeStreamlit:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    @contextmanager
    def form(self, _: str) -> Iterator[FakeStreamlit]:
        yield self

    @contextmanager
    def expander(self, _: str, **__: object) -> Iterator[FakeStreamlit]:
        yield self

    @contextmanager
    def container(self, **__: object) -> Iterator[FakeStreamlit]:
        yield self

    def columns(self, count: int | list[int], **__: object) -> list[FakeStreamlit]:
        return [self for _ in range(count if isinstance(count, int) else len(count))]

    def text_input(self, label: str, **__: object) -> str:
        return self.values.get(label, "")

    def text_area(self, label: str, **__: object) -> str:
        return self.values.get(label, "")

    def selectbox(self, label: str, options: list[str], **__: object) -> str:
        return self.values.get(label, options[0])

    def slider(self, *_: object, **__: object) -> int:
        return 5

    def form_submit_button(self, _: str) -> bool:
        return True

    def button(self, label: str, **__: object) -> bool:
        return label == "Extract"

    def file_uploader(self, *_: object, **__: object) -> None:
        return None

    def __getattr__(self, _: str):
        return lambda *args, **kwargs: None


def _state() -> UiSessionState:
    return UiSessionState(
        settings=UiSettings(mode=UiMode.DEMO),
        resolution=ModeResolution(UiMode.DEMO, "Demo data"),
    )


@pytest.mark.parametrize(
    ("module_name", "value_map"),
    [
        ("search", {"Search query": "appeal deadline"}),
        ("ask", {"Ask a legal question": "هل سأربح هذه الدعوى؟"}),
        (
            "extract",
            {
                "Source mode": "Paste text",
                "Source text": "يلتزم الطرف بالسداد خلال ثلاثين يوماً.",
                "Extraction mode": "Hybrid",
            },
        ),
        ("evaluation", {}),
    ],
)
def test_page_render_functions_are_covered_hermetically(
    module_name: str,
    value_map: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    page = importlib.import_module(f"kawaneen.ui.pages.{module_name}")
    components = importlib.import_module("kawaneen.ui.components")
    styles = importlib.import_module("kawaneen.ui.styles")
    fake = FakeStreamlit(value_map)
    client: UiClient = DemoClient()
    state = _state()
    monkeypatch.setattr(page, "st", fake)
    monkeypatch.setattr(page, "get_context", lambda: (client, state))
    monkeypatch.setattr(components, "st", fake)
    monkeypatch.setattr(styles, "st", fake)

    page.render()


def test_shared_evidence_components_cover_live_degraded_and_demo_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kawaneen.ui.components as components
    import kawaneen.ui.styles as styles

    fake = FakeStreamlit()
    monkeypatch.setattr(components, "st", fake)
    monkeypatch.setattr(styles, "st", fake)

    live = UiSessionState(
        settings=UiSettings(mode=UiMode.LIVE),
        resolution=ModeResolution(UiMode.LIVE, "Live API"),
    )
    degraded = UiSessionState(
        settings=UiSettings(mode=UiMode.AUTO),
        resolution=ModeResolution(None, "Degraded", "offline", True),
    )
    components.render_product_header(live)
    components.render_product_header(degraded)
    assert components.render_status_gate(degraded) is False
    fake.session_state["kawaneen_ui_context"] = (DemoClient(), degraded)
    monkeypatch.setattr(fake, "button", lambda label, **_: label == "Enter portfolio demo mode")
    monkeypatch.setattr(fake, "rerun", lambda: None)
    assert components.render_status_gate(degraded) is False
    components.render_warning_list(["synthetic warning"])

    evidence = DemoClient().search("appeal deadline").results[0]
    components.render_evidence_card(evidence, "appeal")
    components.render_citation_card(DemoClient().answer("appeal deadline").citations[0])
    components.render_mode_note(live)
    components.render_mode_note(_state())
