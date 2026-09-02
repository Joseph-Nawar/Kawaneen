"""Deterministic sanitized SVG figures built from tracked aggregates."""

# SVG attribute strings are intentionally kept literal for deterministic output.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

FIGURE_ROOT = Path("docs/reports/figures/phase15")


def _svg(title: str, labels: list[str], values: list[float], *, suffix: str = "") -> str:
    width, height = 900, 430
    left, bottom, chart_height, chart_width = 190, 340, 250, 650
    minimum = min(min(values, default=0.0), 0.0)
    maximum = max(max(values, default=1.0), 1.0)
    scale = chart_height / (maximum - minimum)
    zero_y = bottom - (0.0 - minimum) * scale
    bars: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        f"<title>{title}</title>",
        '<g fill="none" stroke="currentColor" stroke-width="1">',
        f'<path d="M {left} {zero_y:.1f} H {left + chart_width}"/>',
        f'<path d="M {left} {bottom - chart_height} V {bottom}"/>',
        "</g>",
        f'<text x="{left}" y="32" font-family="sans-serif" font-size="20">{title}</text>',
    ]
    slot = chart_width / max(len(labels), 1)
    bar_width = min(70.0, slot * 0.62)
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = left + slot * index + (slot - bar_width) / 2
        value_y = bottom - (value - minimum) * scale
        y = min(zero_y, value_y)
        bar_height = abs(zero_y - value_y)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="currentColor" opacity="0.72"/>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:.3f}{suffix}</text>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{bottom + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>'
        )
    bars.append("</svg>\n")
    return "\n".join(bars)


def build_report_figures(root: Path) -> tuple[Path, ...]:
    evaluation = root / "data/evaluation"
    phase5 = json.loads((evaluation / "phase5_chunking_metrics.json").read_text(encoding="utf-8"))
    latency = json.loads((evaluation / "phase15_latency_metrics.json").read_text(encoding="utf-8"))
    dialect = json.loads((evaluation / "phase15_dialect_metrics.json").read_text(encoding="utf-8"))
    generator = json.loads(
        (evaluation / "phase15_generator_metrics.json").read_text(encoding="utf-8")
    )
    audit = json.loads((evaluation / "phase15_error_analysis.json").read_text(encoding="utf-8"))
    citation = json.loads(
        (evaluation / "phase15_citation_counterfactual.json").read_text(encoding="utf-8")
    )
    figures: dict[str, str] = {
        "chunking-structure-aware-vs-fixed.svg": _svg(
            "Phase 5 retrieval quality",
            ["fixed-256", "structure"],
            [
                phase5["retrieval_metrics"]["fixed-256-v1"]["ndcg_at_10"],
                phase5["retrieval_metrics"]["legal-structure-v1"]["ndcg_at_10"],
            ],
        ),
        "retrieval-quality-vs-latency.svg": _svg(
            "Phase 15 retrieval quality vs p50 latency",
            list(latency["operations"]),
            [
                float(value["quality"]["nDCG@10"]["mean"])
                for value in latency["operations"].values()
            ],
        ),
        "dialect-msa-paired-effect.svg": _svg(
            "Dialect minus MSA Recall@10",
            ["Egyptian", "Gulf/Saudi", "Levantine", "Pooled"],
            [
                float(
                    dialect["dialects"][name]["hybrid"]["dialect_minus_msa"]["Recall@10"]["delta"]
                )
                for name in ("egyptian", "gulf_saudi", "levantine")
            ]
            + [float(dialect["pooled"]["hybrid"]["dialect_minus_msa"]["Recall@10"]["delta"])],
        ),
        "generator-outcome-support-coverage.svg": _svg(
            "Generator outcome rates",
            ["supported", "coverage", "invalid"],
            [
                float(generator["metrics"]["SupportedAnswerCoverage"]["value"] or 0),
                float(generator["metrics"]["SupportedAnswerCoverage"]["value"] or 0),
                float(generator["metrics"]["invalid_generation_rate"]["value"] or 0),
            ],
        ),
        "automated-diagnostic-composition.svg": _svg(
            "Automated diagnostic composition",
            ["semantic", "lexical", "contract", "borderline"],
            [
                audit["confirmed_failure_taxonomy"].get("semantic retrieval failure", 0),
                audit["confirmed_failure_taxonomy"].get("lexical mismatch", 0),
                audit["non_taxonomy_failure_modes"].get("INVALID_GENERATION_CONTRACT", 0),
                audit["borderline_count"],
            ],
        ),
        "citation-verifier-pre-post.svg": _svg(
            "Citation-verifier defect exposure",
            ["pre", "post"],
            [
                float(citation["pre_defect_surface_rate"]),
                float(citation["post_defect_surface_rate"]),
            ],
        ),
    }
    destination_root = root / FIGURE_ROOT
    destination_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, content in figures.items():
        path = destination_root / filename
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return tuple(paths)
