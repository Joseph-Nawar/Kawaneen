from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_phase14_testing_guide_explains_layers_commands_and_synthetic_policy() -> None:
    guide = (ROOT / "docs/testing.md").read_text(encoding="utf-8")

    for required in (
        "make test-unit",
        "make test-integration",
        "make test-regression",
        "make test-e2e",
        "model cache",
        "Apple Silicon",
        "Qdrant",
        "baseline",
        "Requirement | Layer | Public CI | Model cache | Private data",
    ):
        assert required in guide


def test_phase14_report_and_readme_surface_handoff_artifacts() -> None:
    report = (ROOT / "docs/reports/phase-14-testing-report.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "base SHA" in report
    assert "regression" in readme.lower()
    assert "docs/testing.md" in readme
