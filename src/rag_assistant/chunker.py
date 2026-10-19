"""Split page text into overlapping chunks. A chunk never crosses a page, so page citations are exact."""
from __future__ import annotations

from dataclasses import dataclass

from .loader import Page


@dataclass(frozen=True)
class Chunk:
    id: int  # unique, in document order
    page: int
    text: str


def split_text(text: str, size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)  # prefer to break at a space
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_pages(pages: list[Page], size: int = 800, overlap: int = 150) -> list[Chunk]:
    if size < 50:
        raise ValueError("chunk size must be at least 50 characters")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than the chunk size")
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, size, overlap):
            chunks.append(Chunk(id=len(chunks), page=page.number, text=piece))
    return chunks
