from __future__ import annotations


def test_qualification_report_is_explicitly_local_and_unpublished() -> None:
    from kawaneen.deployment import qualification as phase17_demo_qualify

    report = phase17_demo_qualify.empty_report()
    assert report["provenance"] == "PHASE17_DEV"
    assert report["publication_status"] == "NOT_PUBLISHED_USER_APPROVAL_REQUIRED"
    assert report["qualification_scope"] == "local_constrained_not_huggingface_host"


def test_qualification_requires_all_container_measurements() -> None:
    from kawaneen.deployment.qualification import qualifies_resource_run

    report = {
        "container_image_digest": None,
        "image_size_bytes": 1,
        "startup_time_ms": 1,
        "memory_mb": {"idle_rss": 1, "peak_rss": 1},
        "timings_ms": {
            "search_p50": 1,
            "search_p95": 1,
            "answer_p50": 1,
            "answer_p95": 1,
        },
        "fixed_query_errors": 0,
    }

    assert qualifies_resource_run(report) is False
    report["container_image_digest"] = "sha256:" + "a" * 64
    assert qualifies_resource_run(report) is True
