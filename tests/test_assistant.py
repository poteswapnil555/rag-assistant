import json

import pytest

from conftest import FakeLLM, RaisingLLM, llm_reply
from rag_assistant.assistant import NOT_FOUND, RagAssistant

SICK_Q = "How many sick leave days per year?"


def make(store, llm, **kw):
    return RagAssistant(store, llm, top_k=3, threshold=kw.pop("threshold", 0.3), **kw)


def test_answer_found_with_citation(store):
    llm = FakeLLM(llm_reply("Sick leave is 10 days per year.", ["S1"]))
    a = make(store, llm).ask(SICK_Q)
    assert a.found and "10 days" in a.text and a.reason == "answered"
    assert a.citations and a.citations[0].page == 2
    assert isinstance(a.citations[0].chunk, int) and a.citations[0].snippet
    assert len(llm.prompts) == 1 and "page 2" in llm.prompts[0]


def test_question_not_in_document_never_calls_llm(store):
    llm = FakeLLM(llm_reply("The CEO is Jane Doe.", ["S1"]))  # would be a hallucination if it were used
    a = make(store, llm).ask("Who is the chief executive officer?")
    assert not a.found and NOT_FOUND in a.text.lower() and a.reason == "below_threshold"
    assert a.citations == [] and llm.prompts == []


def test_threshold_is_enforced_in_code_even_if_llm_would_answer(store):
    a = make(store, FakeLLM(llm_reply("invented", ["S1"])), threshold=0.99).ask(SICK_Q)
    assert not a.found and a.reason == "below_threshold"


def test_llm_can_say_not_found(store):
    a = make(store, FakeLLM(llm_reply("", [], found=False))).ask(SICK_Q)
    assert not a.found and NOT_FOUND in a.text.lower() and a.reason == "llm_said_not_found"


def test_answer_without_citations_is_never_returned(store):
    llm = FakeLLM(llm_reply("Sick leave is 10 days.", []))
    a = make(store, llm).ask(SICK_Q)
    assert not a.found and a.reason == "invalid_llm_output" and len(llm.prompts) == 2


def test_citation_to_unknown_source_is_rejected(store):
    a = make(store, FakeLLM(llm_reply("10 days", ["S99"]))).ask(SICK_Q)
    assert not a.found and "S99" in (a.error or "")


def test_invalid_json_then_valid_retries(store):
    llm = FakeLLM("Sure, here you go", llm_reply("Sick leave is 10 days per year.", ["S1"]))
    a = make(store, llm).ask(SICK_Q)
    assert a.found and len(llm.prompts) == 2 and "previous answer was invalid" in llm.prompts[1]


def test_fenced_json_is_accepted(store):
    a = make(store, FakeLLM("```json\n" + llm_reply("10 days", ["S1"]) + "\n```")).ask(SICK_Q)
    assert a.found


def test_found_must_be_a_boolean(store):
    raw = json.dumps({"found": "yes", "answer": "10", "citations": ["S1"]})
    assert not make(store, FakeLLM(raw)).ask(SICK_Q).found


@pytest.mark.parametrize("reply", ["", "null", "[]", "42", "{}"])
def test_empty_or_odd_llm_reply_is_not_an_answer(store, reply):
    a = make(store, FakeLLM(reply)).ask(SICK_Q)
    assert not a.found and a.reason == "invalid_llm_output"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question(store, question):
    llm = FakeLLM("{}")
    a = make(store, llm).ask(question)
    assert not a.found and a.reason == "empty_question" and llm.prompts == []


def test_none_question_does_not_crash(store):
    assert make(store, FakeLLM("{}")).ask(None).reason == "empty_question"


def test_very_long_question_is_refused_without_calls(store):
    llm = FakeLLM("{}")
    a = make(store, llm).ask("sick leave " * 2000)
    assert not a.found and a.reason == "question_too_long" and llm.prompts == []


def test_llm_timeout_is_a_clean_failure(store):
    a = make(store, RaisingLLM(TimeoutError("timed out"))).ask(SICK_Q)
    assert not a.found and a.reason == "llm_error" and "TimeoutError" in a.error and a.citations == []


def test_embedding_failure_is_a_clean_failure(store, monkeypatch):
    monkeypatch.setattr(store, "search", lambda q, k: (_ for _ in ()).throw(RuntimeError("boom")))
    a = make(store, FakeLLM("{}")).ask(SICK_Q)
    assert not a.found and a.reason == "retrieval_error"
