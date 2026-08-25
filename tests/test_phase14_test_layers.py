from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_phase14_markers_are_registered() -> None:
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    for marker in ("integration", "regression", "model_artifact", "e2e", "private_artifact"):
        assert f'"{marker}' in config


def test_phase14_make_targets_are_declared() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in (
        "test-unit",
        "test-integration",
        "test-regression",
        "test-model-regression",
        "test-e2e",
        "test-e2e-private",
        "test-public",
        "check",
    ):
        assert f"{target}:" in makefile


def test_compose_e2e_is_not_part_of_check() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    check_body = makefile.split("check:\n", 1)[1].split("\n\ndoctor:", 1)[0]
    check_body = check_body.split("\n\n", 1)[0]

    assert "test-e2e" not in check_body
