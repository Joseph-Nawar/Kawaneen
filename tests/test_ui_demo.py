from kawaneen.ui.demo import DemoClient


def test_demo_client_contains_arabic_and_english_search_scenarios() -> None:
    client = DemoClient()

    arabic = client.search(query="ما هي مدة الاعتراض؟", limit=8)
    english = client.search(query="appeal deadline", limit=8)

    assert arabic.results[0].text.startswith("تبدأ مدة الاعتراض")
    assert english.results[0].document_title == "Employment Procedures Regulation"
    assert arabic.latency_ms == 0


def test_demo_client_exposes_grounded_and_abstention_answers() -> None:
    client = DemoClient()

    grounded = client.answer(query="ما هي مدة الاعتراض؟")
    abstention = client.answer(query="هل سأربح هذه الدعوى؟")

    assert grounded.answerable is True
    assert grounded.citations[0].quoted_text
    assert abstention.answerable is False
    assert abstention.answer is None
    assert abstention.abstention_reason

