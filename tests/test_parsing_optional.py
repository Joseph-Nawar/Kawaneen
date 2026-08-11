import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kawaneen.parsing.benchmark import preflight_pdfs
from kawaneen.parsing.diagnostics import _result, diagnose_docling
from kawaneen.parsing.docling_backend import DoclingBackend, classify_legal_block
from kawaneen.parsing.health import probe_pdf
from kawaneen.parsing.models import PageHealth


def test_optional_pdf_backends_fail_closed_without_optional_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = __import__("importlib").import_module

    def missing(name: str, *args: object, **kwargs: object) -> object:
        if name in {"pypdf", "docling.document_converter"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", missing)
    with pytest.raises(RuntimeError, match="optional parsing"):
        probe_pdf(tmp_path / "missing.pdf")
    with pytest.raises(RuntimeError, match="Docling"):
        DoclingBackend().parse(tmp_path / "fixture.pdf")


def test_preflight_reports_sanitized_pdf_metadata(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "fixture.pdf"
    path.write_bytes(b"fictional-pdf")

    class FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [SimpleNamespace(mediabox=SimpleNamespace(width=600, height=800))]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))
    monkeypatch.setattr(
        "kawaneen.parsing.benchmark.probe_pdf",
        lambda _path: (PageHealth(page_number=1, text_chars=800, image_count=0),),
    )
    result = preflight_pdfs(tmp_path)
    assert result["schema_version"] == 1
    source = result["sources"][0]
    assert source["page_count"] == 1
    assert source["pages"][0]["likely_layout_complexity"] == "sparse_text"
    assert "text" not in source


def test_docling_diagnostic_distinguishes_usable_output_from_failure() -> None:
    passed = _result(
        device="cpu",
        returncode=0,
        stdout='{"usable_structured_document": true}',
        stderr="",
        elapsed=1.2,
    )
    failed = _result(device="cpu", returncode=-9, stdout="", stderr="killed", elapsed=2.0)
    assert passed["status"] == "passed"
    assert passed["usable_structured_document"] is True
    assert failed["status"] == "failed"
    assert failed["signal"] == 9


def test_docling_diagnostic_captures_subprocess_result(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = '{"usable_structured_document": true}'
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())
    result = diagnose_docling(Path("one-page.pdf"), device="auto")
    assert result["status"] == "passed"
    assert result["device_requested"] == "auto"


def test_docling_backend_uses_explicit_minimal_pipeline(monkeypatch, tmp_path: Path) -> None:
    class Options:
        do_ocr = True
        do_table_structure = True
        heading_hierarchy_options = SimpleNamespace(
            enabled=False, use_numbering=False, use_style=False
        )

    class Converter:
        last = None

        def __init__(self, *, format_options):
            self.format_options = format_options
            Converter.last = self

        def convert(self, _path):
            return SimpleNamespace(
                document=SimpleNamespace(export_to_text=lambda: "fictional structured text")
            )

    fake_base = SimpleNamespace(InputFormat=SimpleNamespace(PDF="pdf"))
    fake_pipeline = SimpleNamespace(PdfPipelineOptions=Options)
    fake_converter = SimpleNamespace(
        DocumentConverter=Converter,
        PdfFormatOption=lambda **kwargs: kwargs,
    )
    original_import = __import__("importlib").import_module

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        modules = {
            "docling.datamodel.base_models": fake_base,
            "docling.datamodel.pipeline_options": fake_pipeline,
            "docling.document_converter": fake_converter,
        }
        if name in modules:
            return modules[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", fake_import)
    monkeypatch.setattr("kawaneen.parsing.docling_backend.version", lambda _: "2.118.1")
    result = DoclingBackend().parse(tmp_path / "fixture.pdf")
    assert result[0].parser_version == "2.118.1"
    assert result[0].extraction_method == "docling_layout_pipeline"
    options = Converter.last.format_options["pdf"]["pipeline_options"]
    assert options.heading_hierarchy_options.enabled is True
    assert options.heading_hierarchy_options.use_numbering is True
    assert options.heading_hierarchy_options.use_style is True


def test_legal_block_classifier_promotes_standalone_arabic_structure_only() -> None:
    assert classify_legal_block("المادة السابعة", "text") == "article_label"
    assert classify_legal_block("الباب الثاني", "text") == "heading"
    assert classify_legal_block("الفصل الأول: أحكام عامة", "text") == "heading"
    assert classify_legal_block("يجوز الرجوع إلى المادة السابعة عند الاقتضاء.", "text") == "text"


def test_docling_backend_emits_geometry_type_and_reading_order(monkeypatch, tmp_path: Path) -> None:
    class Options:
        do_ocr = False
        do_table_structure = False
        heading_hierarchy_options = SimpleNamespace(
            enabled=False, use_numbering=False, use_style=False
        )

    bbox = SimpleNamespace(l=10, b=20, r=110, t=40)
    item = SimpleNamespace(
        text="المادة السابعة",
        label="text",
        prov=[SimpleNamespace(page_no=3, bbox=bbox)],
    )
    document = SimpleNamespace(
        export_to_text=lambda: "المادة السابعة",
        iterate_items=lambda: [(item, 1)],
    )
    converter = SimpleNamespace(
        DocumentConverter=lambda **_kwargs: SimpleNamespace(
            convert=lambda _path: SimpleNamespace(document=document)
        ),
        PdfFormatOption=lambda **kwargs: kwargs,
    )
    modules = {
        "docling.datamodel.base_models": SimpleNamespace(InputFormat=SimpleNamespace(PDF="pdf")),
        "docling.datamodel.pipeline_options": SimpleNamespace(PdfPipelineOptions=Options),
        "docling.document_converter": converter,
    }
    original_import = __import__("importlib").import_module
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: modules[name] if name in modules else original_import(name),
    )
    monkeypatch.setattr("kawaneen.parsing.docling_backend.version", lambda _: "2.118.1")

    result = DoclingBackend().parse(tmp_path / "fixture.pdf")

    assert result[0].block_type == "article_label"
    assert result[0].bounding_box == (10.0, 20.0, 110.0, 40.0)
    assert result[0].page_number == 3
    assert result[0].reading_order == 1
