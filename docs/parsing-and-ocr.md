# PDF parsing and OCR

PDF parsing is a separate optional subsystem. Current CSV and Parquet sources bypass it;
canonical source adapters do not route through PDF or OCR code.

The configured route is:

- healthy embedded text: Docling layout parsing without OCR;
- damaged or mixed text: Docling with selective PDF-aware OCR;
- image-only scan: full-page OCR.

Thresholds live in `configs/parsing/default.toml`. The current OCR candidate is
RapidOCR 3.9.2 using ONNX Runtime, PP-OCRv5, the Arabic mobile recognizer
`arabic_PP-OCRv5_rec_mobile.onnx`, the PP-OCRv5 mobile detector, and the PP-OCRv5
text-line orientation classifier. Model identities, official ModelScope source paths,
and SHA-256 values are recorded in `data/manifests/parsing_benchmark.json`. OCR model
downloads are never automatic in CI or ordinary checks. Parser, OCR engine/model,
page, bounding-box, block type, reading order, and extraction method are retained as
provenance.

The optional parsing group uses pypdf for the lightweight embedded-text health probe
and Docling for layout-aware parsing. PyMuPDF is intentionally not included because
its MuPDF licensing requires a separate AGPL/commercial review. Docling and the
selected OCR runtime/model licences must be reviewed before distribution.

The private benchmark targets about 30 representative pages. It calculates CER, WER,
heading precision/recall/F1, exact and semantic article-number accuracy, reading-order
accuracy, and page-reference preservation. The project thresholds are provisional
gates, not universal standards; no critical article-number error may remain in a
manually approved core page set.

## Current qualification status

The benchmark manifest now records `geometry_benchmark_validated_routes_measured`.
Three authorized PDFs were preflighted and the frozen 30 real pages contain 102
anchored regions. Anchored Gold v2 uses PDFium text geometry for born-digital MOJ
and Umm Al-Qura pages and 21 independent visual SAMA annotations; the reviewed
text remains marked `human_verified=false`. The deterministic contact sheet and
private audit confirm 102 valid in-bounds boxes and unchanged source hashes.

Metric scopes are frozen in `data/manifests/parsing_benchmark.json`: 3 MOJ pages for
legal structure, 7 SAMA pages for legal OCR, 10 Umm Al-Qura regulatory pages for legal
layout, 3 general complex-layout pages, and 7 non-legal complex-layout pages. The
general and non-legal pages are excluded from legal article-label denominators.
Scoped empirical metrics are recorded as external-AI diagnostics, not human-gold
qualification scores. `legal_structure` passes CER/WER/article-number/reading-order
thresholds but fails the current heading heuristic (F1 0.333). `legal_ocr` fails
CER 9.535%, WER 27.511%, and heading F1 0.662 against the scanned-page gates.
`legal_regulatory_layout` fails CER 18.688%, WER 37.980%, heading F1 0.350, semantic
article accuracy 20.0%, and reading order 71.667%. General and non-legal pages remain
diagnostic-only and excluded from legal denominators.
The 45 controlled variants (18 clean/low-resolution and 27 deterministic degraded
scans) were completed with a 30-second per-variant watchdog. All 45 produced
non-empty Arabic output, but no CER/WER or structural score is reported from them
because mapped variant gold is unavailable.

## Dependency and model licensing

| Component | Licence basis | Release treatment |
| --- | --- | --- |
| `pypdf` | BSD-3-Clause | Optional embedded-text probe |
| Docling | MIT | Optional layout pipeline; review transitive dependencies |
| `rapidocr` 3.9.2 | Apache-2.0 | Optional OCR runtime |
| `onnxruntime` 1.28.0 | MIT | Optional ONNX Runtime backend |
| `python-bidi` 0.6.11 | LGPL-3.0-or-later | Optional Arabic display-order dependency |
| Arabic PP-OCRv5 weights | ModelScope/RapidOCR artifact; exact weight terms remain separately unresolved | No weights bundled; ignored local cache only |

Package/runtime licensing does not establish permission to redistribute model
weights. PyMuPDF is not a direct dependency; any transitive Docling dependency
must be reviewed before deployment.

## Remediation evidence

The previous OCR result was invalid because `rapidocr_onnxruntime` 1.4.4 used
bundled Chinese PP-OCRv4 artifacts. It is no longer an optional dependency. The
replacement smoke test used RapidOCR 3.9.2, ONNX Runtime 1.28.0, and the explicit
Arabic PP-OCRv5 model set; it produced 36 non-empty regions containing Arabic code
points in 1.383 seconds. The seven image-only pages produced Arabic output in the
150-DPI diagnostic batch, averaging 20.714 regions and 0.907 seconds/page.

The previous Docling termination was not reproduced once the pipeline was made
explicit: disabling OCR and table enrichment and selecting pypdfium2 yielded a
usable one-page structured document on CPU (5.760 seconds) and auto, which selected
CPU (5.705 seconds). The exact sanitized subprocess records are in
`data/manifests/parsing_diagnostics.json`. The earlier bare-converter behavior is
therefore attributed to default pipeline/model initialization rather than a proven
intrinsic PDF conversion defect. MPS was not exercised because auto selected CPU.

The final geometry-based qualification covered the frozen development/holdout split
(21/9 pages). Development-only selection chose Docling for MOJ, RapidOCR full-page
200 DPI for SAMA, and Docling for Umm Al-Qura. Each holdout route was evaluated once:
all three routes failed at least one unchanged quality gate and are therefore
`manual_review_or_abstain`; no route is qualified for automatic v1 ingestion.
Per-route development and holdout metrics are in
`docs/reports/phase-03-parsing-qualification-report.md` and the private result
artifact.

## Closure evaluation guard

The historical external metrics used full-page concatenation for regions that do not
share a prediction identity or geometry. They are retained as diagnostic history, not
qualification evidence. The evaluator now matches stable region identities first and
uses geometry only for anonymous blocks; unrelated page text cannot be compared as a
single region. Docling now emits block type, bounding box, and reading order, with
conservative Arabic standalone detection for article, chapter, and part headings.

The historical full-page diagnostic remains retained for provenance only. The valid
evaluator assigns predicted blocks to anchored gold boxes by geometry and never uses
text similarity to establish correspondence; unrelated same-page regions cannot be
concatenated. Failed optional PDF/OCR routes require manual review or abstention,
with no additional Phase 3 tuning cycle. Deployment still requires a separate review
of PP-OCRv5 weight terms and optional Docling dependency licensing.
