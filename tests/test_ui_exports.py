import json

from kawaneen.ui.demo import DemoClient
from kawaneen.ui.exports import extraction_csv, extraction_json


def test_extraction_json_preserves_segment_identity() -> None:
    response = DemoClient().extract("يلتزم الطرف بالسداد خلال ثلاثين يوماً.")

    payload = json.loads(extraction_json((("segment-001", response),)))

    assert payload[0]["segment_id"] == "segment-001"
    assert payload[0]["response"]["result"]["schema_version"] == "phase11-extraction-v1"


def test_extraction_csv_is_flattened_and_downloadable() -> None:
    response = DemoClient().extract("The party must pay within thirty days.")

    csv_text = extraction_csv((("segment-001", response),)).decode()

    assert "segment_id" in csv_text
    assert "field" in csv_text
    assert "obligations" in csv_text
