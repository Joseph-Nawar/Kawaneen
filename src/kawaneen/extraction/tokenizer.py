"""Pinned local-only tokenizer configuration for future hybrid execution."""

from kawaneen.generation.tokenizer import LazyHuggingFaceTokenizer

QWEN_TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_TOKENIZER_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def pinned_local_tokenizer() -> LazyHuggingFaceTokenizer:
    return LazyHuggingFaceTokenizer(
        identity=QWEN_TOKENIZER_ID,
        revision=QWEN_TOKENIZER_REVISION,
    )
