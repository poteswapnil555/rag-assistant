"""Usage: python -m rag_assistant [--pdf FILE] ["question"]   (no question -> interactive prompt)"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .assistant import RagAssistant
from .chunker import chunk_pages
from .config import env_int
from .embeddings import SentenceTransformerEmbedder
from .index import VectorStore
from .llm import OpenAICompatibleClient
from .loader import PdfError, load_pdf

DEFAULT_PDF = Path("data/sample/NIST.AI.100-1.pdf")


def show(answer) -> None:
    print(answer.text)
    for c in answer.citations:
        print(f"  [page {c.page}, chunk {c.chunk}] (similarity {c.score}) {c.snippet[:100]}...")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="?", help="ask one question and exit")
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = ap.parse_args(argv)

    try:
        llm = OpenAICompatibleClient()  # checks the environment first, before the slow steps
        chunks = chunk_pages(load_pdf(args.pdf), env_int("CHUNK_SIZE", 800), env_int("CHUNK_OVERLAP", 150))
        store = VectorStore(SentenceTransformerEmbedder(), chunks)
    except (RuntimeError, ValueError, PdfError, ImportError, OSError) as e:
        print(f"Cannot start: {e}", file=sys.stderr)
        if not args.pdf.exists() and args.pdf == DEFAULT_PDF:
            print("Hint: python scripts/download_pdf.py", file=sys.stderr)
        return 2

    assistant = RagAssistant(store, llm)
    if args.question is not None:
        show(assistant.ask(args.question))
        return 0
    print(f"Loaded {args.pdf.name}: {len(chunks)} chunks. Empty line to quit.")
    while True:
        try:
            q = input("> ")
        except EOFError:
            return 0
        if not q.strip():
            return 0
        show(assistant.ask(q))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
