from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "src/kawaneen/ui/app.py")


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=10).run()


def test_demo_search_page_renders_evidence_and_supports_english_query(monkeypatch) -> None:
    monkeypatch.setenv("KAWANEEN_UI_MODE", "demo")
    app = _run()

    assert app.title[0].value == "Search"
    assert any("demo data" in item.value.lower() for item in app.info)

    app.text_input[0].set_value("appeal deadline")
    app.button[0].click().run()

    assert any("Employment Procedures Regulation" in item.value for item in app.markdown)


def test_demo_ask_page_renders_grounded_answer_and_abstention(monkeypatch) -> None:
    monkeypatch.setenv("KAWANEEN_UI_MODE", "demo")
    app = _run()
    app.switch_page("pages/ask.py").run()

    assert any("Grounded answer" in item.value for item in app.markdown)
    app.text_area[0].set_value("هل سأربح هذه الدعوى؟")
    app.button[0].click().run()

    assert any("Grounded answer not issued" in item.value for item in app.markdown)


def test_demo_extract_page_shows_experimental_label_and_downloads(monkeypatch) -> None:
    monkeypatch.setenv("KAWANEEN_UI_MODE", "demo")
    app = _run()
    app.switch_page("pages/extract.py").run()

    app.text_area[0].set_value("يلتزم الطرف بالسداد خلال ثلاثين يوماً.")
    app.selectbox[1].select("Hybrid")
    app.button[0].click().run()

    assert any("PHASE11_HYBRID_EXPERIMENTAL_LIMITED" in item.value for item in app.warning)
    assert len(app.download_button) == 2


def test_demo_evaluation_page_shows_provenance_and_latency_label(monkeypatch) -> None:
    monkeypatch.setenv("KAWANEEN_UI_MODE", "demo")
    app = _run()
    app.switch_page("pages/evaluation.py").run()

    assert any("Live API readiness" in item.value for item in app.markdown)
    assert any("Live session latency — not a benchmark" in item.value for item in app.markdown)
    assert any("source hashes" in item.value.lower() for item in app.markdown)
