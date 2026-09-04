from kawaneen.ui.demo import DemoClient


def test_demo_client_contains_arabic_and_english_search_scenarios() -> None:
    client = DemoClient()

    arabic = client.search(query="ما هي مدة الاعتراض؟", limit=8)
    english = client.search(query="appeal deadline", limit=8)

    assert arabic.results[0].text.startswith("تبدأ مدة الاعتراض")
    assert english.results[0].document_title == "Employment Procedures Regulation"
    assert arabic.jurisdiction == english.jurisdiction == "KAWANEEN_DEMO"
    assert arabic.latency_ms == 0


def test_demo_client_exposes_grounded_and_abstention_answers() -> None:
    client = DemoClient()

    grounded = client.answer(query="ما هي مدة الاعتراض؟")
    abstention = client.answer(query="هل سأربح هذه الدعوى؟")

    assert grounded.answerable is True
    assert grounded.jurisdiction == "KAWANEEN_DEMO"
    assert grounded.citations[0].quoted_text
    assert abstention.answerable is False
    assert abstention.answer is None
    assert abstention.abstention_reason
    assert abstention.jurisdiction == "KAWANEEN_DEMO"


def test_demo_client_exposes_demo_jurisdiction_for_extraction_and_documents() -> None:
    client = DemoClient()

    extraction = client.extract("يلتزم الطرف بالسداد خلال ثلاثين يوماً.")
    documents = client.list_documents()
    detail = client.get_document("demo-procedure")

    assert extraction.result.jurisdiction == "KAWANEEN_DEMO"
    assert all(item.jurisdiction == "KAWANEEN_DEMO" for item in documents.items)
    assert detail.document.jurisdiction == "KAWANEEN_DEMO"
