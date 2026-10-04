"""Text embeddings. Runs locally via fastembed (ONNX runtime, no PyTorch)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from codebase_archaeologist.config import Settings
from codebase_archaeologist.log import get_logger

log = get_logger(__name__)


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """bge-small-en-v1.5 (384-d, quantized ONNX, ~67 MB) on CPU.

    bge retrieves better when short *queries* carry an instruction prefix;
    documents are embedded as-is.
    """

    def __init__(self, settings: Settings) -> None:
        from fastembed import TextEmbedding

        self._model_name = settings.embedding_model
        self._query_prefix = settings.embedding_query_prefix
        self._batch_size = settings.embedding_batch_size
        log.info("Loading embedding model %s", self._model_name)
        self._model = TextEmbedding(model_name=self._model_name)
        self._dim = len(next(iter(self._model.embed(["dimension probe"]))))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.embed(texts, batch_size=self._batch_size)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([self._query_prefix + text]))).tolist()


class HashEmbedder:
    """Deterministic bag-of-words embedder for tests: no model download, and texts
    sharing words get similar vectors, so retrieval ordering is meaningful."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"hash-{self._dim}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in re.findall(r"[a-z0-9_]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
