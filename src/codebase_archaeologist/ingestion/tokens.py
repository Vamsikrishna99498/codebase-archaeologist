"""Token counting with the embedding model's own tokenizer, so chunk sizes match
exactly what the embedder sees (bge-small truncates beyond 512 tokens)."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

TokenCounter = Callable[[str], int]


@lru_cache
def load_token_counter(model_name: str) -> TokenCounter:
    """Download (once, then cached by Hugging Face) the tokenizer for `model_name`."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer.no_truncation()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count
