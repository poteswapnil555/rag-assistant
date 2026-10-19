"""Evaluate the assistant on a question set (answerable and NOT answerable questions).

  python -m rag_assistant.evaluate --mock   # plumbing check: fake embeddings + fake LLM, synthetic PDF. NOT accuracy.
  python -m rag_assistant.evaluate --live   # real local embeddings + real LLM from environment variables
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .assistant import Answer, RagAssistant
from .chunker import chunk_pages
from .config import env_float, env_int
from .index import VectorStore
from .loader import PdfError, load_pdf

NIST_PDF = Path("data/sample/NIST.AI.100-1.pdf")
NIST_QUESTIONS = Path("eval/questions_nist.json")
SYNTHETIC_QUESTIONS = Path("eval/questions_synthetic.json")
MOCK_THRESHOLD = 0.2  # the hashing embedder has a different score scale than the real model


def load_questions(path: Path) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read questions file {path}: {e}") from e
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list of questions")
    for q in data:
        if not isinstance(q, dict) or not q.get("id") or not q.get("question") or not isinstance(q.get("answerable"), bool):
            raise ValueError(f"{path}: every question needs id, question and answerable (true/false): {q}")
        if q["answerable"] and not q.get("expected_keywords"):
            raise ValueError(f"{path}: answerable question {q['id']} needs expected_keywords")
    return data


def keywords_present(text: str, keywords: list[str]) -> bool:
    """Every keyword must appear; 'a|b' means either a or b."""
    low = text.lower()
    return all(any(alt.strip() in low for alt in kw.lower().split("|")) for kw in keywords)


@dataclass
class Report:
    mode: str
    answerable: int = 0
    answered_correctly: int = 0
    citation_correct: int = 0
    false_refusals: int = 0
    wrong_answers: int = 0
    unanswerable: int = 0
    correctly_refused: int = 0
    wrongly_answered: int = 0
    llm_errors: int = 0
    details: list[dict] = field(default_factory=list)

    def add(self, q: dict, a: Answer) -> None:
        error = a.reason in {"llm_error", "retrieval_error"}
        correct = False
        cite_ok = None
        if q["answerable"]:
            self.answerable += 1
            if error:
                self.llm_errors += 1
            elif not a.found:
                self.false_refusals += 1
            elif keywords_present(a.text, q["expected_keywords"]):
                self.answered_correctly += 1
                correct = True
            else:
                self.wrong_answers += 1
            if a.found and q.get("expected_pages"):
                cite_ok = any(c.page in q["expected_pages"] for c in a.citations)
                self.citation_correct += cite_ok
        else:
            self.unanswerable += 1
            if error:
                self.llm_errors += 1
            elif not a.found:
                self.correctly_refused += 1
                correct = True
            else:
                self.wrongly_answered += 1
        self.details.append({
            "id": q["id"], "question": q["question"], "answerable": q["answerable"], "correct": correct,
            "found": a.found, "answer": a.text, "reason": a.reason, "top_score": a.top_score,
            "expected_keywords": q.get("expected_keywords"), "expected_pages": q.get("expected_pages"),
            "citations": [{"page": c.page, "chunk": c.chunk, "score": c.score} for c in a.citations],
            "citation_on_expected_page": cite_ok, "error": a.error,
        })

    def summary(self) -> dict:
        total = self.answerable + self.unanswerable
        return {
            "mode": self.mode,
            "questions": total,
            "overall_accuracy": round((self.answered_correctly + self.correctly_refused) / max(total, 1), 4),
            "answerable_questions": self.answerable,
            "answered_correctly": self.answered_correctly,
            "citation_on_expected_page": f"{self.citation_correct}/{self.answerable}",
            "false_refusals": self.false_refusals,
            "wrong_answers": self.wrong_answers,
            "unanswerable_questions": self.unanswerable,
            "correctly_refused": self.correctly_refused,
            "wrongly_answered": self.wrongly_answered,
            "llm_or_retrieval_errors": self.llm_errors,
        }


class MockLLM:
    """Extractive stand-in for the LLM: returns the text of source S1 with a citation. Plumbing only."""

    def complete(self, system: str, user: str) -> str:
        m = re.search(r"\[S1\] \(page \d+, chunk \d+\)\n(.*?)(?:\n\n\[S2\]|\n\nQuestion:)", user, re.DOTALL)
        if not m:
            return json.dumps({"found": False, "answer": "", "citations": []})
        return json.dumps({"found": True, "answer": m.group(1).strip(), "citations": ["S1"]})


def evaluate(assistant: RagAssistant, questions: list[dict], mode: str, delay: float = 0.0,
             sleep=time.sleep, progress=None) -> Report:
    """delay: seconds to pause between questions (rate-limited free tiers). progress: callable(done, total)."""
    report = Report(mode=mode)
    for i, q in enumerate(questions, start=1):
        report.add(q, assistant.ask(q["question"]))
        if progress:
            progress(i, len(questions))
        if delay > 0 and i < len(questions):
            sleep(delay)
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="no key, no downloads; NOT an accuracy measurement")
    group.add_argument("--live", action="store_true", help="real embeddings + the LLM configured by environment variables")
    ap.add_argument("--pdf", type=Path, help="default: synthetic PDF for --mock, the NIST PDF for --live")
    ap.add_argument("--questions", type=Path, help="default: eval/questions_synthetic.json (mock) or questions_nist.json (live)")
    ap.add_argument("--threshold", type=float, help="similarity threshold (default SIMILARITY_THRESHOLD or 0.35)")
    ap.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                    help="pause between questions, e.g. 13 for a 5 requests/minute limit")
    ap.add_argument("--out", type=Path, help="save the summary plus per-question details as JSON")
    args = ap.parse_args(argv)

    try:
        if args.mock:
            from .embeddings import HashEmbedder
            from .synthetic import make_handbook_pdf

            mode, embedder, llm = "mock", HashEmbedder(), MockLLM()
            pdf = args.pdf or make_handbook_pdf(Path(tempfile.mkdtemp()) / "handbook.pdf")
            qpath = args.questions or SYNTHETIC_QUESTIONS
            threshold = args.threshold if args.threshold is not None else MOCK_THRESHOLD
            print("MODE: mock -- plumbing check with fake embeddings and a fake LLM. These numbers are NOT accuracy.")
        else:
            from .embeddings import SentenceTransformerEmbedder
            from .llm import OpenAICompatibleClient

            pdf = args.pdf or NIST_PDF
            if not pdf.exists() and pdf == NIST_PDF:
                print(f"{pdf} not found. Run: python scripts/download_pdf.py", file=sys.stderr)
                return 2
            llm = OpenAICompatibleClient()  # fail on missing env before loading the embedding model
            embedder = SentenceTransformerEmbedder()
            mode, qpath = "live", args.questions or NIST_QUESTIONS
            threshold = args.threshold if args.threshold is not None else env_float("SIMILARITY_THRESHOLD", 0.35)
            print("MODE: live -- local embeddings and the real LLM from environment variables.")
        chunks = chunk_pages(load_pdf(pdf), env_int("CHUNK_SIZE", 800), env_int("CHUNK_OVERLAP", 150))
        questions = load_questions(qpath)
        store = VectorStore(embedder, chunks)
    except (RuntimeError, ValueError, PdfError, ImportError, OSError) as e:
        print(f"Cannot run evaluation: {e}", file=sys.stderr)
        return 2

    assistant = RagAssistant(store, llm, threshold=threshold)

    def show(done: int, total: int) -> None:
        print(f"{done}/{total} done", file=sys.stderr, flush=True)

    report = evaluate(assistant, questions, mode, delay=args.delay, progress=show)
    summary = report.summary()
    summary.update({"pdf": pdf.name, "chunks": len(chunks), "threshold": threshold})
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({**summary, "details": report.details}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
