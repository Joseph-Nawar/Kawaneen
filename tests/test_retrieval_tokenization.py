from kawaneen.normalization.tokenization import tokenize
from kawaneen.retrieval.tokenization import represent, tokenize_retrieval


def test_retrieval_tokenizer_is_the_phase4_tokenizer() -> None:
    text = f"المادة {chr(0x661)}، (أ)"
    assert tokenize_retrieval(text) == tokenize(text)


def test_raw_and_light_representations_only_derive_search_text() -> None:
    original = "أـب  test"

    raw = represent(original, "arabic-raw-v1")
    light = represent(original, "arabic-light-v1")

    assert raw.display_text == original
    assert light.display_text == original
    assert raw.search_text == "أـب test"
    assert light.search_text == "اب test"
