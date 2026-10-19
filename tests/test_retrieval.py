import numpy as np
import pytest

from rag_assistant.chunker import Chunk, chunk_pages, split_text
from rag_assistant.embeddings import HashEmbedder
from rag_assistant.index import VectorStore
from rag_assistant.loader import NoTextLayerError, Page, PdfError, load_pdf
from rag_assistant.synthetic import make_image_only_pdf


# ---- loader ------------------------------------------------------------------
def test_load_pdf_returns_pages_with_numbers(handbook_pdf):
    pages = load_pdf(handbook_pdf)
    assert [p.number for p in pages] == [1, 2, 3]
    assert "3 days per week" in pages[0].text


def test_empty_file(tmp_path):
    f = tmp_path / "empty.pdf"
    f.write_bytes(b"")
    with pytest.raises(PdfError, match="empty"):
        load_pdf(f)


def test_missing_file(tmp_path):
    with pytest.raises(PdfError, match="cannot read"):
        load_pdf(tmp_path / "nope.pdf")


def test_corrupt_pdf(tmp_path):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.4 definitely not a pdf \x00\x01")
    with pytest.raises(PdfError, match="not a valid PDF"):
        load_pdf(f)


def test_scanned_pdf_without_text_layer(tmp_path):
    f = make_image_only_pdf(tmp_path / "scan.pdf")
    with pytest.raises(NoTextLayerError, match="no text layer"):
        load_pdf(f)


def test_very_large_file_rejected(handbook_pdf, monkeypatch):
    monkeypatch.setenv("MAX_FILE_MB", "0.0001")
    with pytest.raises(PdfError, match="MAX_FILE_MB"):
        load_pdf(handbook_pdf)


# ---- chunker -----------------------------------------------------------------
def test_chunks_keep_page_numbers_and_unique_ids(chunks):
    assert [c.id for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2, 3}


def test_chunk_never_crosses_pages():
    pages = [Page(1, "alpha " * 100), Page(2, "beta " * 100)]
    for c in chunk_pages(pages, size=120, overlap=20):
        assert ("alpha" in c.text) != ("beta" in c.text)


def test_chunk_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(200))
    pieces = split_text(text, size=100, overlap=30)
    assert len(pieces) > 3 and all(len(p) <= 100 for p in pieces)
    assert pieces[0].split()[-1] in pieces[1]  # overlap: end of one chunk reappears in the next


def test_chunk_text_is_fully_covered():
    text = " ".join(f"w{i}" for i in range(300))
    joined = " ".join(split_text(text, size=90, overlap=20))
    assert all(f"w{i}" in joined.split() for i in range(300))


def test_long_unbroken_text_terminates():
    assert len(split_text("x" * 10_000, size=100, overlap=99)) > 1


@pytest.mark.parametrize("size,overlap", [(10, 0), (100, 100), (100, -1)])
def test_bad_chunk_settings_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text " * 50)], size=size, overlap=overlap)


# ---- embeddings + FAISS --------------------------------------------------------
def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder()
    a, b = e.embed(["remote work days"]), e.embed(["remote work days"])
    assert np.array_equal(a, b) and abs(np.linalg.norm(a[0]) - 1) < 1e-5


def test_search_finds_the_right_page(store):
    top = store.search("How many sick leave days per year?", k=1)[0]
    assert top.chunk.page == 2 and top.score > 0.3


def test_unrelated_query_scores_low(store):
    assert store.search("Who is the chief executive officer?", k=1)[0].score < 0.2


def test_k_larger_than_index_is_ok(store):
    assert len(store.search("laptop", k=1000)) == len(store)


def test_empty_chunk_list_rejected():
    with pytest.raises(ValueError, match="zero chunks"):
        VectorStore(HashEmbedder(), [])


def test_bad_embedder_output_rejected():
    class Nan:
        def embed(self, texts):
            return np.full((len(texts), 4), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="invalid vectors"):
        VectorStore(Nan(), [Chunk(0, 1, "text")])
