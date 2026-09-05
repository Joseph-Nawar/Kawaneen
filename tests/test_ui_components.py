from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from kawaneen.ui.config import ModeResolution, UiMode, UiSettings
from kawaneen.ui.state import UiSessionState


class _HeaderStreamlit:
    def __init__(self) -> None:
        self.html_blocks: list[str] = []
        self.warnings: list[str] = []

    def __enter__(self) -> _HeaderStreamlit:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def html(self, body: str) -> None:
        self.html_blocks.append(body)

    @contextmanager
    def container(self, **_: object) -> Iterator[_HeaderStreamlit]:
        yield self

    def columns(self, count: int, **_: object) -> list[_HeaderStreamlit]:
        return [self for _ in range(count)]

    def page_link(self, *_: object, **__: object) -> None:
        return None

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def test_public_demo_header_is_truthful_and_keeps_safety_boundary(monkeypatch) -> None:
    import kawaneen.ui.components as components
    import kawaneen.ui.styles as styles

    fake = _HeaderStreamlit()
    monkeypatch.setattr(components, "st", fake)
    monkeypatch.setattr(styles, "st", fake)
    state = UiSessionState(
        settings=UiSettings(mode=UiMode.LIVE, public_demo=True),
        resolution=ModeResolution(UiMode.LIVE, "Live API"),
    )

    components.render_product_header(state)

    rendered = "\n".join(fake.html_blocks + fake.warnings)
    assert "Arabic Legal Intelligence · Synthetic public demo" in rendered
    assert "Arabic Legal Intelligence · Saudi Arabia" not in rendered
    for required_text in (
        "PUBLIC DEMO",
        "Fictional/synthetic curated corpus",
        "Reduced retrieval profile",
        "No generative legal answer",
        "Not real Saudi legislation",
        "Not legal advice",
    ):
        assert required_text in rendered
    assert not fake.warnings
