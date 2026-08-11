from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kawaneen.parsing.ocr import (
    ArabicOCRConfig,
    ArabicOCRConfigurationError,
    assert_arabic_model,
    build_rapidocr,
)


def test_arabic_configuration_is_explicit() -> None:
    config = ArabicOCRConfig()
    assert_arabic_model(config)
    assert config.language == "arabic"
    assert config.ocr_version == "PP-OCRv5"


def test_rapidocr_builder_passes_explicit_arabic_parameters(monkeypatch) -> None:
    captured = {}

    class FakeRapidOCR:
        def __init__(self, *, params):
            captured.update(params)

    fake = SimpleNamespace(
        EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"),
        LangDet=SimpleNamespace(CH="ch"),
        LangRec=SimpleNamespace(ARABIC="arabic"),
        ModelType=SimpleNamespace(MOBILE="mobile"),
        OCRVersion=SimpleNamespace(PPOCRV5="PP-OCRv5"),
        RapidOCR=FakeRapidOCR,
    )
    monkeypatch.setitem(__import__("sys").modules, "rapidocr", fake)
    build_rapidocr(ArabicOCRConfig(), model_dir=Path("models"))
    assert captured["Rec.lang_type"] == "arabic"
    assert captured["Rec.ocr_version"] == "PP-OCRv5"
    assert captured["Global.model_root_dir"] == "models"


def test_previous_chinese_default_cannot_qualify_arabic() -> None:
    config = ArabicOCRConfig(recognizer_model="ch_PP-OCRv4_rec_infer.onnx")
    with pytest.raises(ArabicOCRConfigurationError, match="Arabic"):
        assert_arabic_model(config)


@pytest.mark.parametrize("field", ["engine", "ocr_version", "language", "model_type"])
def test_arabic_gate_rejects_wrong_runtime_selection(field: str) -> None:
    values = {field: "wrong"}
    with pytest.raises(ArabicOCRConfigurationError):
        assert_arabic_model(ArabicOCRConfig(**values))
