"""FAISS vector store (cosine similarity via inner product on normalised vectors)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunker import Chunk
from .embeddings import Embedder


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, about -1..1; higher is more similar


class VectorStore:
    def __init__(self, embedder: Embedder, chunks: list[Chunk]) -> None:
        import faiss

        if not chunks:
            raise ValueError("cannot build an index from zero chunks")
        vectors = np.ascontiguousarray(embedder.embed([c.text for c in chunks]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not np.isfinite(vectors).all():
            raise ValueError("embedder returned invalid vectors")
        self._embedder = embedder
        self._chunks = chunks
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        k = max(1, min(k, len(self._chunks)))
        q = np.ascontiguousarray(self._embedder.embed([query]), dtype=np.float32)
        if not np.isfinite(q).all():
            raise ValueError("embedder returned invalid vectors for the query")
        scores, ids = self._index.search(q, k)
        return [Hit(self._chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
