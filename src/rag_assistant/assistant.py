"""Question answering with citations. "Not found" is enforced in code, not only in the prompt.

Order of checks for every question:
1. empty or too-long question -> refused, no retrieval, no LLM call
2. best FAISS similarity below the threshold -> "not found in the document", the LLM is NOT called
3. the LLM must return JSON with found + answer + citations; citations must be among the retrieved chunks
4. found=false, missing citations, invalid JSON (after one retry) or an LLM error -> never an invented answer
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import env_float, env_int
from .index import Hit, VectorStore
from .llm import LLMClient

NOT_FOUND = "not found in the document"
NOT_FOUND_TEXT = "Not found in the document."

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.
Rules:
- Use only facts stated in the sources. Never use outside knowledge and never guess.
- If the sources do not contain the answer, set "found" to false.
- Cite every source you used by its id (for example "S1").
- Return ONLY one JSON object with exactly these keys:
  {"found": true or false, "answer": "short answer text", "citations": ["S1", "S2"]}"""


@dataclass(frozen=True)
class Citation:
    page: int
    chunk: int
    snippet: str
    score: float


@dataclass
class Answer:
    text: str
    found: bool
    citations: list[Citation] = field(default_factory=list)
    reason: str = "answered"
    top_score: float | None = None
    error: str | None = None


def _refuse(reason: str, text: str = NOT_FOUND_TEXT, top_score: float | None = None, error: str | None = None) -> Answer:
    return Answer(text=text, found=False, reason=reason, top_score=top_score, error=error)


def parse_reply(raw: str) -> dict:
    raw = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    data = json.loads(fenced.group(1) if fenced else raw)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")
    return data


class RagAssistant:
    def __init__(
        self,
        store: VectorStore,
        llm: LLMClient,
        top_k: int | None = None,
        threshold: float | None = None,
        max_question_chars: int | None = None,
        max_retries: int = 1,
    ) -> None:
        self.store = store
        self.llm = llm
        self.top_k = top_k if top_k is not None else env_int("TOP_K", 4)
        self.threshold = threshold if threshold is not None else env_float("SIMILARITY_THRESHOLD", 0.35)
        self.max_question_chars = max_question_chars if max_question_chars is not None else env_int("MAX_QUESTION_CHARS", 1000)
        self.max_retries = max_retries

    def ask(self, question: str) -> Answer:
        q = (question or "").strip()
        if not q:
            return _refuse("empty_question", "Please enter a question.")
        if len(q) > self.max_question_chars:
            return _refuse("question_too_long", f"The question is too long (limit {self.max_question_chars} characters).")

        try:
            hits = self.store.search(q, self.top_k)
        except Exception as e:  # embedding failure etc.
            return _refuse("retrieval_error", f"Search failed: {type(e).__name__}", error=str(e)[:300])
        top_score = hits[0].score if hits else None
        usable = [h for h in hits if h.score >= self.threshold]
        if not usable:
            return _refuse("below_threshold", top_score=top_score)

        sources = {f"S{i}": h for i, h in enumerate(usable, start=1)}
        prompt = self._prompt(q, sources)
        last_problem = ""
        for _ in range(self.max_retries + 1):
            user = prompt if not last_problem else f"{prompt}\n\nYour previous answer was invalid: {last_problem}\nReturn corrected JSON only."
            try:
                raw = self.llm.complete(SYSTEM_PROMPT, user)
            except Exception as e:  # timeout, connection error, exhausted rate limit...
                msg = f"{type(e).__name__}: {str(e)[:300]}"
                return _refuse("llm_error", f"Could not get an answer from the language model ({type(e).__name__}).",
                               top_score, error=msg)
            try:
                data = parse_reply(raw)
            except ValueError as e:  # includes JSONDecodeError
                last_problem = f"not valid JSON ({str(e)[:100]})"
                continue
            problem = self._check(data, sources)
            if problem:
                last_problem = problem
                continue
            if data["found"] is False:
                return _refuse("llm_said_not_found", top_score=top_score)
            cites = [Citation(h.chunk.page, h.chunk.id, h.chunk.text[:200], round(h.score, 4))
                     for h in (sources[c] for c in dict.fromkeys(data["citations"]))]
            return Answer(text=data["answer"].strip(), found=True, citations=cites, top_score=top_score)
        return _refuse("invalid_llm_output", top_score=top_score, error=last_problem)

    @staticmethod
    def _prompt(question: str, sources: dict[str, Hit]) -> str:
        blocks = [f"[{sid}] (page {h.chunk.page}, chunk {h.chunk.id})\n{h.chunk.text}" for sid, h in sources.items()]
        return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"

    @staticmethod
    def _check(data: dict, sources: dict[str, Hit]) -> str:
        """Return a description of what is wrong with the model's JSON, or '' if it is acceptable."""
        found = data.get("found")
        if not isinstance(found, bool):
            return '"found" must be true or false'
        if not found:
            return ""
        answer = data.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return '"answer" must be a non-empty string when found is true'
        cites = data.get("citations")
        if not isinstance(cites, list) or not cites:
            return '"citations" must list at least one source id when found is true'
        unknown = [c for c in cites if not isinstance(c, str) or c not in sources]
        if unknown:
            return f"unknown citation ids {unknown}; use only {list(sources)}"
        return ""
