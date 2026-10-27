"""embeddings.py

Text embedders for the article index.

Responsibilities:
- OllamaEmbedder: calls /api/embed of a local Ollama server. It refuses any base_url that is not a loopback address,
  so an article server never talks to the network beyond this machine.
- HashingEmbedder: a deterministic bag-of-words embedder with no model behind it, for tests and for running without
  Ollama. Its vectors are only good enough to exercise the pipeline.
- make_embedder: builds one from the "embedding" block of a tool's config.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
WORD = re.compile(r"[a-z0-9]+")


class EmbeddingUnavailable(RuntimeError):
    """The embedding service cannot be reached or refused the request."""


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout_s: float = 60.0, batch_size: int = 16):
        host = urlparse(base_url).hostname
        if host not in LOOPBACK_HOSTS:
            raise ValueError(f"Embedding base_url {base_url!r} is not a loopback address; article servers stay on this machine.")
        self.name = f"ollama:{model}"
        self.model = model
        self.url = base_url.rstrip("/") + "/api/embed"
        self.timeout_s = timeout_s
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                response = httpx.post(self.url, json={"model": self.model, "input": batch}, timeout=self.timeout_s)
            except httpx.HTTPError as exc:
                raise EmbeddingUnavailable(f"Ollama is not reachable at {self.url}: {exc}") from exc
            body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            if response.status_code != 200 or "embeddings" not in body:
                raise EmbeddingUnavailable(f"Ollama refused to embed with {self.model!r}: {body.get('error', response.text[:200])}")
            vectors.extend(body["embeddings"])
        return normalise(np.asarray(vectors, dtype=np.float32))


class HashingEmbedder:
    """Signed feature hashing of words and word pairs; deterministic and dependency-free."""

    def __init__(self, dims: int = 256):
        self.name = f"hash:{dims}"
        self.dims = dims

    def _index(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dims, (1.0 if value >> 63 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            words = WORD.findall(text.lower())
            for token in words + [f"{a}_{b}" for a, b in zip(words, words[1:])]:
                column, sign = self._index(token)
                matrix[row, column] += sign / math.sqrt(1 + len(token) / 8)
        return normalise(matrix)


def make_embedder(settings: dict[str, Any]) -> Embedder:
    provider = settings.get("provider", "ollama")
    if provider == "ollama":
        return OllamaEmbedder(settings["model"], settings.get("base_url", "http://localhost:11434"), settings.get("timeout_s", 60))
    if provider == "hash":
        return HashingEmbedder(int(settings.get("dims", 256)))
    raise ValueError(f"Unknown embedding provider {provider!r}; use 'ollama' or 'hash'.")
