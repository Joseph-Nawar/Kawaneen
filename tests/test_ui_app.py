from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import kawaneen.ui.app as app

APP = str(Path(__file__).parents[1] / "src/kawaneen/ui/app.py")


class _Navigation:
    def __init__(self) -> None:
        self.ran = False

    def run(self) -> None:
        self.ran = True


class _Column:
    def __enter__(self) -> _Column:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _Container(_Column):
    pass


class _AppStreamlit:
    def __init__(self) -> None:
        self.config: dict[str, object] = {}
        self.pages: list[object] = []
        self.page_links: list[tuple[object, str | None]] = []
        self.navigation_position: str | None = None
        self.navigation_result = _Navigation()

    def set_page_config(self, **kwargs: object) -> None:
        self.config = kwargs

    def Page(self, path: str, **kwargs: object) -> object:
        page = SimpleNamespace(path=path, **kwargs)
        self.pages.append(page)
        return page

    def columns(self, spec: int | list[int], **_: object) -> list[_Column]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_Column() for _ in range(count)]

    def container(self, **_: object) -> _Container:
        return _Container()

    def page_link(self, page: object, *, label: str | None = None, **_: object) -> None:
        self.page_links.append((page, label))

    def navigation(self, pages: list[object], **kwargs: object) -> _Navigation:
        assert pages == self.pages
        self.navigation_position = str(kwargs["position"])
        return self.navigation_result


def test_main_configures_and_runs_the_four_page_navigation(monkeypatch) -> None:
    fake = _AppStreamlit()
    monkeypatch.setattr(app, "st", fake)

    app.main()

    assert fake.config["layout"] == "wide"
    assert [page.title for page in fake.pages] == [
        "Search",
        "Ask",
        "Extract",
        "Evaluation",
    ]
    assert all(not getattr(page, "icon", None) for page in fake.pages)
    assert fake.navigation_result.ran is True


def test_main_hides_builtin_navigation(monkeypatch) -> None:
    fake = _AppStreamlit()
    monkeypatch.setattr(app, "st", fake)

    app.main()

    assert fake.navigation_position == "hidden"


def test_product_navigation_uses_existing_page_paths_and_labels(monkeypatch) -> None:
    import kawaneen.ui.components as components

    fake = _AppStreamlit()
    monkeypatch.setattr(components, "st", fake)

    components.render_product_navigation("Search")

    assert fake.page_links == [
        ("pages/search.py", "Search"),
        ("pages/ask.py", "Ask"),
        ("pages/extract.py", "Extract"),
        ("pages/evaluation.py", "Evaluation"),
    ]
    assert all("(current)" not in label for _, label in fake.page_links if label is not None)


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=10).run()


def test_demo_search_page_renders_evidence_and_supports_english_query(monkeypatch) -> None:
    monkeypatch.setenv("KAWANEEN_UI_MODE", "demo")
    app = _run()

    assert any("<h1>Search</h1>" in item.proto.body for item in app.get("html"))
    assert any("demo data" in item.value.lower() for item in app.info)

    app.text_input[0].set_value("appeal deadline")
    app.button[0].click().run()

    assert any("Employment Procedures Regulation" in item.proto.body for item in app.get("html"))


def test_demo_ask_page_renders_grounded_answer_and_abstention(monkeypatch) -> None:
    monkeypatch.setenv("KAWANEEN_UI_MODE", "demo")
    app = _run()
    app.switch_page("pages/ask.py").run()

    assert any("Grounded answer" in item.value for item in app.markdown)
    app.text_area[0].set_value("هل سأربح هذه الدعوى؟")
    app.button[0].click().run()

    assert any("Grounded answer not issued" in item.proto.body for item in app.get("html"))


def test_demo_extract_page_shows_experimental_label_and_downloads(monkeypatch) -> None:
    monkeypatch.setenv("KAWANEEN_UI_MODE", "demo")
    app = _run()
    app.switch_page("pages/extract.py").run()

    app.text_area[0].set_value("يلتزم الطرف بالسداد خلال ثلاثين يوماً.")
    app.selectbox[1].select("Hybrid")
    app.button[0].click().run()

    assert any("PHASE11_HYBRID_EXPERIMENTAL_LIMITED" in item.value for item in app.warning)
    assert len(app.download_button) == 2


def test_demo_extract_corpus_mode_shows_paginated_document_bounds(monkeypatch) -> None:
    monkeypatch.setenv("KAWANEEN_UI_MODE", "demo")
    app = _run()
    app.switch_page("pages/extract.py").run()

    app.selectbox[0].select("Corpus document").run()

    assert any("Documents 1\u20132 of 2" in item.value for item in app.caption)


def test_demo_evaluation_page_shows_provenance_and_latency_label(monkeypatch) -> None:
    monkeypatch.setenv("KAWANEEN_UI_MODE", "demo")
    app = _run()
    app.switch_page("pages/evaluation.py").run()

    assert any("Model capability snapshot" in item.value for item in app.markdown)
    assert not any("Live API readiness" in item.value for item in app.markdown)
    assert not any("live readiness" in item.value for item in app.caption)
    assert any("Live session latency — not a benchmark" in item.value for item in app.markdown)
    assert any("Technical provenance" in item.label for item in app.expander)
    assert any("BM25 + BGE-M3" in item.value for item in app.caption)
    assert any("Search" in item.value and "Answer" in item.value for item in app.caption)
