# Anchored Benchmark Gold v2 Design

## Goal

Close the final Phase 3 benchmark blocker by producing an independently anchored,
geometry-based qualification benchmark for the frozen 30 selected pages and 102
externally reviewed regions, without changing canonical/statutory data or entering
Phase 4.

## Architecture

The private benchmark pipeline has four boundaries. `anchored_gold.py` converts the
existing externally verified region text into immutable top-left page coordinates:
born-digital pages use PDFium text rectangles and fail closed on ambiguous spans;
image-only SAMA pages use a deterministic private annotation path and preserve the
existing transcriptions without using parser output. `benchmark.py` receives gold
regions and parser blocks in the same coordinate system and performs spatial,
typed, geometry-only correspondence. `qualification.py` freezes a deterministic
source-stratified development/holdout split, selects only bounded existing route
configurations on development pages, and evaluates holdout pages exactly once.
`audit.py` validates integrity, renders the private contact sheet, and emits the
final support matrix/gate report.

## Data flow and safeguards

The existing selection manifest, integrated external gold, source PDFs, and parser
configuration candidates are inputs. The private v2 JSONL stores exact reviewed
text, source hashes, page dimensions, top-left boxes, coordinate provenance, and
review provenance. Gold text is copied only from the existing independent review;
gold geometry is either PDFium-derived from the source text layer or independently
visually/manual anchored for SAMA. Prediction output is allowed only in a separate
evaluation result and never participates in gold construction.

Every page and region is validated before scoring: 30 pages, 102 unique regions,
non-empty in-bounds boxes, unchanged hashes, valid coordinate metadata, no ambiguous
born-digital anchors, and no prediction-derived provenance. A private contact sheet
overlays every box over deterministic page renders.

## Evaluation

Gold boxes are matched to predicted blocks on page and typed geometry, using documented
overlap/center rules and geometric/declared reading order for multi-block text. Text
is concatenated only from blocks assigned to that gold box. Headings and article labels
are scored from typed spatial matches, and reading order compares anchored gold IDs
with the predicted block sequence. Synthetic tests prevent same-page unrelated regions
from being compared or concatenated.

The split is deterministic and source-stratified across MOJ legal structure, SAMA legal
OCR, and Umm Al-Qura regulatory layout. Development pages choose among already
implemented bounded DPI/preprocessing/layout candidates. Configuration and split are
then frozen; each holdout page is evaluated once. Existing thresholds remain unchanged.

## Failure policy

SAMA visual anchoring is independent of RapidOCR/Docling. If automated visual placement
cannot be reliably established, the private annotation utility is used and records
`independent_ai_visual_anchor` with `human_verified=false`; if boxes remain uncertain,
gold generation fails closed. A route that misses quality thresholds is reported as
`manual_review_or_abstain`, not retuned. Only a route meeting existing thresholds is
eligible for automatic v1 ingestion.

## Verification

Tests cover coordinate conversion, scaling, multi-line union, ambiguous anchoring,
page/hash mismatch, invalid boxes, spatial block assignment, typed structure metrics,
reading order, split freezing, and one-time holdout evaluation. Final handoff runs
pytest, Ruff, Pyright, pre-commit, `make check`, deterministic reruns, and git diff/
staging checks. No private gold/PDF/model artifacts are staged.
