from __future__ import annotations


def test_qualification_report_is_explicitly_local_and_unpublished() -> None:
    from kawaneen.deployment import qualification as phase17_demo_qualify

    report = phase17_demo_qualify.empty_report()
    assert report["provenance"] == "PHASE17_DEV"
    assert report["publication_status"] == "NOT_PUBLISHED_USER_APPROVAL_REQUIRED"
    assert report["qualification_scope"] == "local_constrained_not_huggingface_host"
