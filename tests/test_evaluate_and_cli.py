import importlib.util
import json
from pathlib import Path

import pytest

from conftest import FakeLLM, llm_reply
from rag_assistant import __main__ as cli
from rag_assistant.assistant import RagAssistant
from rag_assistant.evaluate import (MockLLM, evaluate, keywords_present, load_questions, main)
from rag_assistant.synthetic import make_image_only_pdf

ROOT = Path(__file__).resolve().parent.parent
ENV_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_TENANT_ID",
            "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]


@pytest.fixture
def no_env(monkeypatch):
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


# ---- question files ---------------------------------------------------------------
@pytest.mark.parametrize("name,expected_total", [("questions_nist.json", 16), ("questions_synthetic.json", 14)])
def test_question_files_are_valid_and_mixed(name, expected_total):
    qs = load_questions(ROOT / "eval" / name)
    assert len(qs) == expected_total
    assert any(not q["answerable"] for q in qs) and any(q["answerable"] for q in qs)
    assert len({q["id"] for q in qs}) == len(qs)


def test_bad_question_file_is_rejected(tmp_path):
    f = tmp_path / "q.json"
    f.write_text(json.dumps([{"id": "x", "question": "q", "answerable": True}]))
    with pytest.raises(ValueError, match="expected_keywords"):
        load_questions(f)
    f.write_text("not json")
    with pytest.raises(ValueError, match="cannot read"):
        load_questions(f)


def test_keyword_matching_supports_alternatives():
    assert keywords_present("The system must STOP.", ["cease|stop"])
    assert not keywords_present("nothing", ["cease|stop"])
    assert keywords_present("a b", ["a", "b"]) and not keywords_present("a", ["a", "b"])


# ---- evaluation ----------------------------------------------------------------------
def _qs():
    return load_questions(ROOT / "eval" / "questions_synthetic.json")


def test_mock_evaluation_plumbing(store):
    report = evaluate(RagAssistant(store, MockLLM(), top_k=3, threshold=0.2), _qs(), "mock")
    s = report.summary()
    assert s["questions"] == 14 and s["answerable_questions"] == 10 and s["unanswerable_questions"] == 4
    assert len(report.details) == 14


def test_unanswerable_questions_must_be_refused_to_count(store):
    # An LLM that always claims it found something, with the similarity gate wide open
    llm = FakeLLM(llm_reply("The CEO is Jane.", ["S1"]))
    report = evaluate(RagAssistant(store, llm, top_k=3, threshold=-1.0), _qs(), "mock")
    assert report.wrongly_answered == 4 and report.correctly_refused == 0


def test_false_refusals_are_counted(store):
    report = evaluate(RagAssistant(store, FakeLLM(llm_reply("", [], found=False)), top_k=3, threshold=0.2), _qs(), "mock")
    assert report.false_refusals == 10 and report.correctly_refused == 4


def test_llm_errors_are_counted_separately(store):
    class Down:
        def complete(self, s, u):
            raise TimeoutError("down")

    report = evaluate(RagAssistant(store, Down(), top_k=3, threshold=0.2), _qs(), "mock")
    assert report.llm_errors == 10 and report.answered_correctly == 0


def test_delay_and_progress(store):
    slept, seen = [], []
    evaluate(RagAssistant(store, MockLLM(), threshold=0.2), _qs(), "mock", delay=1.5, sleep=slept.append,
             progress=lambda d, t: seen.append((d, t)))
    assert seen[0] == (1, 14) and seen[-1] == (14, 14) and slept == [1.5] * 13


def test_cli_mock_prints_mode_and_never_says_accuracy(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main(["--mock"]) == 0
    cap = capsys.readouterr()
    assert "MODE: mock" in cap.out and "NOT accuracy" in cap.out and "14/14 done" in cap.err


def test_cli_saves_details(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "e.json"
    assert main(["--mock", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["mode"] == "mock" and len(data["details"]) == 14
    assert {"id", "question", "answer", "reason", "citations", "correct"} <= set(data["details"][0])


def test_live_without_env_fails_with_clear_message(no_env, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main(["--live", "--pdf", str(ROOT / "README.md")]) == 2
    assert "No LLM configured" in capsys.readouterr().err


def test_live_without_nist_pdf_points_to_download_script(no_env, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no data/sample/NIST.AI.100-1.pdf here
    assert main(["--live"]) == 2
    assert "download_pdf.py" in capsys.readouterr().err


def test_evaluation_on_scanned_pdf_fails_cleanly(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    pdf = make_image_only_pdf(tmp_path / "scan.pdf")
    assert main(["--mock", "--pdf", str(pdf)]) == 2
    assert "no text layer" in capsys.readouterr().err


# ---- question-answering CLI ---------------------------------------------------------------
def test_ask_cli_without_env_prints_clear_message(no_env, capsys, handbook_pdf):
    assert cli.main(["--pdf", str(handbook_pdf), "How many sick days?"]) == 2
    assert "No LLM configured" in capsys.readouterr().err


def test_ask_cli_with_missing_model_variable(no_env, capsys, handbook_pdf):
    no_env.setenv("OPENAI_API_KEY", "fake")
    assert cli.main(["--pdf", str(handbook_pdf), "q"]) == 2
    assert "OPENAI_MODEL is not set" in capsys.readouterr().err


def test_ask_cli_with_missing_pdf_hints_download(no_env, capsys, tmp_path, monkeypatch):
    no_env.setenv("OPENAI_API_KEY", "fake")
    no_env.setenv("OPENAI_MODEL", "m")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["question"]) == 2
    assert "download_pdf.py" in capsys.readouterr().err


def test_ask_cli_prints_answer_and_citation(monkeypatch, capsys, handbook_pdf):
    from rag_assistant.embeddings import HashEmbedder

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.2")
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda: FakeLLM(llm_reply("10 days per year.", ["S1"])))
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", HashEmbedder)  # no model download in tests
    assert cli.main(["--pdf", str(handbook_pdf), "How many sick leave days per year?"]) == 0
    out = capsys.readouterr().out
    assert "10 days per year." in out and "[page 2, chunk" in out


# ---- download script (no network) ---------------------------------------------------------------
def _script():
    spec = importlib.util.spec_from_file_location("download_pdf", ROOT / "scripts" / "download_pdf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Resp:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self.data


def test_download_saves_valid_pdf(tmp_path, monkeypatch):
    mod = _script()
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=0: _Resp(b"%PDF-1.7 fake"))
    dest = tmp_path / "x" / "doc.pdf"
    assert mod.download("https://example.invalid/a.pdf", dest) == 0 and dest.read_bytes().startswith(b"%PDF")


def test_download_rejects_non_pdf_and_keeps_nothing(tmp_path, monkeypatch, capsys):
    mod = _script()
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=0: _Resp(b"<html>nope</html>"))
    dest = tmp_path / "doc.pdf"
    assert mod.download("https://example.invalid/a.pdf", dest) == 1 and not dest.exists()
    assert "not a PDF" in capsys.readouterr().err


def test_download_network_error_is_clean(tmp_path, monkeypatch, capsys):
    mod = _script()

    def boom(req, timeout=0):
        raise mod.urllib.error.URLError("offline")

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    assert mod.download("https://example.invalid/a.pdf", tmp_path / "d.pdf") == 1
    assert "Download failed" in capsys.readouterr().err


def test_download_skips_existing_file(tmp_path, monkeypatch):
    mod = _script()
    dest = tmp_path / "d.pdf"
    dest.write_bytes(b"%PDF old")
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not download"))
    assert mod.download("https://example.invalid/a.pdf", dest) == 0 and dest.read_bytes() == b"%PDF old"
