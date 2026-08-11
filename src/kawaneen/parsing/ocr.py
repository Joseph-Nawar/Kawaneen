"""Explicit, lazy RapidOCR configuration for Arabic qualification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ArabicOCRConfigurationError(ValueError):
    """Raised when a non-Arabic model could be used for Arabic qualification."""


@dataclass(frozen=True)
class ArabicOCRConfig:
    """Model-selection contract recorded by the parser benchmark."""

    engine: str = "onnxruntime"
    ocr_version: str = "PP-OCRv5"
    language: str = "arabic"
    model_type: str = "mobile"
    detector_model: str = "ch_PP-OCRv5_det_mobile.onnx"
    recognizer_model: str = "arabic_PP-OCRv5_rec_mobile.onnx"
    classifier_model: str = "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx"
    detector_source_url: str = (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
        "onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx"
    )
    recognizer_source_url: str = (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
        "onnx/PP-OCRv5/rec/arabic_PP-OCRv5_rec_mobile.onnx"
    )
    classifier_source_url: str = (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
        "onnx/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx"
    )
    detector_sha256: str = "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"
    recognizer_sha256: str = "c1192e632d0baa9146ae5b756a0e635e3dc63c1733737ebfd1629e87144e9295"
    classifier_sha256: str = "54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7"


def assert_arabic_model(config: ArabicOCRConfig) -> None:
    """Reject configurations that do not explicitly select Arabic PP-OCRv5."""

    if config.engine != "onnxruntime":
        raise ArabicOCRConfigurationError("Arabic qualification requires ONNX Runtime")
    if config.ocr_version != "PP-OCRv5":
        raise ArabicOCRConfigurationError("Arabic qualification requires PP-OCRv5")
    if config.language != "arabic":
        raise ArabicOCRConfigurationError("Arabic qualification requires language=arabic")
    if config.model_type != "mobile":
        raise ArabicOCRConfigurationError("Arabic qualification requires the mobile model")
    recognizer = config.recognizer_model.lower()
    if "arabic" not in recognizer:
        raise ArabicOCRConfigurationError(
            "Arabic qualification requires an explicitly Arabic recognizer artifact"
        )
    if "ch_ppocrv4" in recognizer or "ch_" in recognizer:
        raise ArabicOCRConfigurationError(
            "Chinese/default recognizer artifacts cannot qualify Arabic OCR"
        )


def build_rapidocr(config: ArabicOCRConfig, *, model_dir: Path | None = None) -> Any:
    """Build RapidOCR only when optional dependencies and explicit models are available."""

    assert_arabic_model(config)
    try:
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
    except ImportError as exc:
        raise RuntimeError("install the optional parsing dependency group") from exc

    params: dict[str, Any] = {
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.ARABIC,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Cls.model_type": ModelType.MOBILE,
        "Cls.ocr_version": OCRVersion.PPOCRV5,
    }
    if model_dir is not None:
        params["Global.model_root_dir"] = str(model_dir)
    return RapidOCR(params=params)
