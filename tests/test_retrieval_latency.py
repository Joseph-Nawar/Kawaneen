import pytest

from kawaneen.retrieval.latency import LatencySummary


def test_latency_summary_reports_percentiles_and_metadata() -> None:
    summary = LatencySummary.from_samples(
        (1.0, 2.0, 3.0, 4.0, 5.0), device="cpu", package_versions={"numpy": "x"}, threads=1
    )

    assert summary.count == 5
    assert summary.p50 == 3.0
    assert summary.p95 == 5.0
    assert summary.device == "cpu"


def test_measure_latency_warms_up_and_times_all_queries() -> None:
    from kawaneen.retrieval.latency import measure_latency

    calls: list[str] = []

    def search(query: str) -> None:
        calls.append(query)

    summary = measure_latency(search, ("a", "b", "c"), warmup_count=1)

    assert summary.count == 3
    assert calls == ["a", "a", "b", "c"]


def test_latency_validation_and_observation_capture() -> None:
    from kawaneen.retrieval.latency import measure_latency

    with pytest.raises(ValueError, match="non-empty"):
        LatencySummary.from_samples((), device="cpu", package_versions={}, threads=1)
    with pytest.raises(TypeError, match="callable"):
        measure_latency(None, ("q",))
    observations: list[float] = []
    measure_latency(lambda _query: None, ("q",), warmup_count=0, observations=observations)
    assert len(observations) == 1 and observations[0] >= 0
