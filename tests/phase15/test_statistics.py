from __future__ import annotations

import pytest

from kawaneen.phase15.statistics import (
    cohens_kappa,
    paired_bootstrap_delta,
    paired_rank_biserial,
    paired_risk_difference,
)


def test_bootstrap_is_deterministic_and_reports_pair_counts() -> None:
    left = (1.0, 2.0, 4.0, 5.0)
    right = (0.0, 3.0, 2.0, 5.0)
    first = paired_bootstrap_delta(left, right, seed=20260826)
    second = paired_bootstrap_delta(left, right, seed=20260826)

    assert first == second
    assert first.delta == pytest.approx(0.5)
    assert (first.wins, first.ties, first.losses) == (2, 1, 1)
    assert first.replicates == 2000


def test_bootstrap_rejects_empty_or_mismatched_pairs() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_delta((), ())
    with pytest.raises(ValueError):
        paired_bootstrap_delta((1.0,), (1.0, 2.0))


def test_rank_biserial_handles_ties_and_all_wins() -> None:
    assert paired_rank_biserial((1.0, 2.0), (1.0, 2.0)) == 0.0
    assert paired_rank_biserial((2.0, 3.0), (1.0, 2.0)) == 1.0


def test_paired_binary_risk_difference_and_kappa() -> None:
    result = paired_risk_difference((1, 1, 0, 0), (0, 1, 0, 0), seed=20260826)
    assert result.risk_difference == pytest.approx(-0.25)
    assert result.discordant_pairs == {"before_positive_after_negative": 1, "before_negative_after_positive": 0}
    assert cohens_kappa(("a", "b", "a"), ("a", "b", "a")) == 1.0


def test_kappa_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError):
        cohens_kappa(("a",), ("b", "a"))
