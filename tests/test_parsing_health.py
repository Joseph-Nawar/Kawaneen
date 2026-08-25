from __future__ import annotations

from types import SimpleNamespace

import pytest

import kawaneen.parsing.health as health


class _Resource:
    def __init__(self, subtype: str) -> None:
        self.subtype = subtype

    def get_object(self) -> dict[str, str]:
        return {"/Subtype": self.subtype}


class _Page:
    def __init__(self, text: str | None, resources: object) -> None:
        self.text = text
        self.resources = resources

    def extract_text(self) -> str | None:
        return self.text

    def get(self, key: str, default: object) -> object:
        return self.resources if key == "/Resources" else default


def test_probe_pdf_reports_text_and_images(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    pages = (
        _Page(
            "page text",
            {"/XObject": {"image": _Resource("/Image"), "other": _Resource("/Form")}},
        ),
        _Page(None, None),
    )
    fake_pypdf = SimpleNamespace(PdfReader=lambda _path: SimpleNamespace(pages=pages))
    monkeypatch.setattr(health.importlib, "import_module", lambda _name: fake_pypdf)

    result = health.probe_pdf(tmp_path / "sample.pdf")

    assert result[0].page_number == 1
    assert result[0].text_chars == len("page text")
    assert result[0].image_count == 1
    assert result[1].page_number == 2
    assert result[1].text_chars == 0
    assert result[1].image_count == 0


def test_probe_pdf_reports_missing_optional_dependency(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    def missing(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(health.importlib, "import_module", missing)

    with pytest.raises(RuntimeError, match="optional parsing dependencies"):
        health.probe_pdf(tmp_path / "sample.pdf")
