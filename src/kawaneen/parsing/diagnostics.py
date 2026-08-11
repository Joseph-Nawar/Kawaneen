"""Subprocess diagnostics for the optional Docling layout boundary."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_CHILD = r"""
import json
import sys
from importlib.metadata import version
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

options = PdfPipelineOptions()
options.do_ocr = False
options.do_table_structure = False
print(json.dumps({
    "phase": "models",
    "docling": version("docling"),
    "docling_core": version("docling-core"),
    "pypdfium2": version("pypdfium2"),
    "ocr_enabled": options.do_ocr,
    "table_structure_enabled": options.do_table_structure,
    "model_artifact_check": "layout_only_options_created",
}, sort_keys=True), flush=True)
converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=options)
})
print(json.dumps({
    "phase": "converter_ready",
    "backend": "pypdfium2",
    "device": sys.argv[2],
}), flush=True)
result = converter.convert(sys.argv[1])
text = result.document.export_to_text()
print(json.dumps({
    "phase": "converted",
    "page_count": len(result.document.pages),
    "usable_structured_document": bool(text.strip()),
}), flush=True)
"""


def diagnose_docling(
    path: Path, *, device: str = "cpu", timeout_seconds: int = 90
) -> dict[str, Any]:
    """Run one-page layout parsing in a child process and return sanitized diagnostics."""

    started = time.perf_counter()
    command = [sys.executable, "-c", _CHILD, str(path), device]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return _result(
            device=device,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed=time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return _result(
            device=device,
            returncode=None,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or ""),
            elapsed=time.perf_counter() - started,
            timeout_seconds=timeout_seconds,
        )


def _result(
    *,
    device: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
    elapsed: float,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backend": "pypdfium2",
        "device_requested": device,
        "returncode": returncode,
        "signal": -returncode if returncode is not None and returncode < 0 else None,
        "status": "passed" if returncode == 0 else "failed",
        "usable_structured_document": returncode == 0
        and '"usable_structured_document": true' in stdout,
        "elapsed_seconds": round(elapsed, 3),
        "timeout_seconds": timeout_seconds,
        "stdout": stdout[-12000:],
        "stderr": stderr[-12000:],
        "peak_memory_mb": None,
    }
