import json

import pytest

from rag_assistant.chunker import chunk_pages
from rag_assistant.embeddings import HashEmbedder
from rag_assistant.index import VectorStore
from rag_assistant.loader import load_pdf
from rag_assistant.synthetic import make_handbook_pdf


class FakeLLM:
    """Returns queued replies in order (the last one repeats); records prompts."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class RaisingLLM:
    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        raise self.exc


def llm_reply(answer: str, citations: list[str], found: bool = True) -> str:
    return json.dumps({"found": found, "answer": answer, "citations": citations})


@pytest.fixture(scope="session")
def handbook_pdf(tmp_path_factory):
    return make_handbook_pdf(tmp_path_factory.mktemp("pdf") / "handbook.pdf")


@pytest.fixture(scope="session")
def chunks(handbook_pdf):
    return chunk_pages(load_pdf(handbook_pdf), size=400, overlap=60)


@pytest.fixture(scope="session")
def store(chunks):
    return VectorStore(HashEmbedder(), chunks)
