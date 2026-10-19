"""Embedders. The real one is local (sentence-transformers). HashEmbedder is for tests and --mock only."""
from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np

from .config import env

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array (len(texts), dim) of L2-normalised vectors."""
        ...


class SentenceTransformerEmbedder:
    """Local embeddings: no key, no quota. The model is downloaded once on first use."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or env("EMBED_MODEL", DEFAULT_MODEL)
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


_STOPWORDS = frozenset(
    "a an the of to in on at for and or is are was were be been by with as it its this that these those from "
    "what which who whom how when where why do does did can could should would will may might must not no "
    "if then than so such per into out up about over under any all each".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOPWORDS:
            continue
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        tokens.append(t)
    return tokens


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. No model, no download. For tests and --mock only;
    it measures word overlap, not meaning."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                out[row, zlib.crc32(tok.encode()) % self.dim] += 1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out
