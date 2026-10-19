"""PDF loading. Every failure is a PdfError with a clear message."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import env_float


class PdfError(Exception):
    """The PDF cannot be used (missing, empty, corrupt, too large, encrypted...)."""


class NoTextLayerError(PdfError):
    """The PDF opened fine but has no extractable text (e.g. scanned pages). OCR is not supported."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based page number in the PDF
    text: str


def max_file_bytes() -> int:
    return int(env_float("MAX_FILE_MB", 50.0) * 1024 * 1024)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_pdf(path: Path | str) -> list[Page]:
    """Return the pages that have text. Raises PdfError / NoTextLayerError."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as e:
        raise PdfError(f"cannot read file: {e}") from e
    if size == 0:
        raise PdfError(f"{path.name}: file is empty")
    if size > max_file_bytes():
        raise PdfError(f"{path.name}: file is {size / 1024 / 1024:.1f} MB, over the MAX_FILE_MB limit")

    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfError(f"{path.name}: PDF is password protected")
        raw_pages = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    except PdfError:
        raise
    except Exception as e:
        raise PdfError(f"{path.name}: not a valid PDF ({type(e).__name__}: {e})") from e

    if not raw_pages:
        raise PdfError(f"{path.name}: PDF has no pages")
    pages = [Page(i, clean(t)) for i, t in raw_pages if clean(t)]
    if not pages:
        raise NoTextLayerError(
            f"{path.name}: PDF has no text layer (scanned or image-only). OCR is not supported in this project."
        )
    return pages
