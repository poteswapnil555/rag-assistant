"""LLM client: mode selection, missing env, rate limit retry. SDKs are replaced by fakes; no key or network."""
import openai
import pytest

from rag_assistant import llm as llm_mod
from rag_assistant.assistant import RagAssistant
from rag_assistant.config import azure_auth_mode, llm_configured, llm_provider

ALL = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION",
       "AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "OPENAI_API_KEY", "OPENAI_BASE_URL",
       "OPENAI_MODEL", "LLM_TIMEOUT_SECONDS"]


@pytest.fixture
def no_env(monkeypatch):
    for v in ALL:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


# ---- missing environment variables ---------------------------------------------------
def test_no_llm_configured_has_clear_message(no_env):
    assert not llm_configured()
    with pytest.raises(RuntimeError, match="No LLM configured"):
        llm_mod.OpenAICompatibleClient()


def test_missing_openai_model_has_clear_message(no_env):
    no_env.setenv("OPENAI_API_KEY", "fake")
    with pytest.raises(RuntimeError, match="OPENAI_MODEL is not set"):
        llm_mod.OpenAICompatibleClient()


def test_azure_without_deployment_has_clear_message(no_env):
    no_env.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid/")
    no_env.setenv("AZURE_OPENAI_API_KEY", "fake")
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_DEPLOYMENT is not set"):
        llm_mod.OpenAICompatibleClient()


def test_bad_timeout_has_clear_message(no_env):
    no_env.setenv("OPENAI_API_KEY", "fake")
    no_env.setenv("OPENAI_MODEL", "m")
    no_env.setenv("LLM_TIMEOUT_SECONDS", "soon")
    with pytest.raises(RuntimeError, match="LLM_TIMEOUT_SECONDS must be a number"):
        llm_mod.OpenAICompatibleClient()


# ---- mode selection -------------------------------------------------------------------
def test_openai_compatible_mode(no_env):
    seen = {}
    no_env.setenv("OPENAI_API_KEY", "fake")
    no_env.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    no_env.setenv("OPENAI_MODEL", "some-model")
    no_env.setenv("LLM_TIMEOUT_SECONDS", "7")
    no_env.setattr(openai, "OpenAI", lambda **kw: seen.update(kw) or object())
    c = llm_mod.OpenAICompatibleClient()
    assert llm_provider() == "openai" and c._model == "some-model"
    assert seen["base_url"] == "https://example.invalid/v1" and seen["timeout"] == 7.0


def _azure(no_env):
    captured = {}
    no_env.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid/")
    no_env.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")
    no_env.setattr(openai, "AzureOpenAI", lambda **kw: captured.update(kw) or object())
    no_env.setattr(llm_mod, "build_token_provider", lambda: "TOKEN_PROVIDER")
    return captured


def test_azure_key_mode(no_env):
    cap = _azure(no_env)
    no_env.setenv("AZURE_OPENAI_API_KEY", "fake")
    llm_mod.OpenAICompatibleClient()
    assert azure_auth_mode() == "key" and cap["api_key"] == "fake" and "azure_ad_token_provider" not in cap


def test_azure_token_mode_when_no_key(no_env):
    cap = _azure(no_env)
    for k in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"):
        no_env.setenv(k, "x")
    llm_mod.OpenAICompatibleClient()
    assert azure_auth_mode() == "token" and cap["azure_ad_token_provider"] == "TOKEN_PROVIDER" and "api_key" not in cap


# ---- rate limit (429) --------------------------------------------------------------------
class Boom(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


class _Message:
    content = '{"found": true, "answer": "10 days", "citations": ["S1"]}'


class _Choice:
    message = _Message()


class _Resp:
    choices = [_Choice()]


class FlakyClient:
    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return _Resp()


@pytest.fixture
def waits(monkeypatch):
    w = []
    monkeypatch.setattr(llm_mod, "_sleep", w.append)
    return w


def _client(fake):
    c = object.__new__(llm_mod.OpenAICompatibleClient)
    c._client, c._model = fake, "m"
    return c


def test_429_waits_for_time_in_error_then_succeeds(waits):
    fake = FlakyClient(Boom("quota exceeded. Please retry in 12.5s.", 429))
    assert "10 days" in _client(fake).complete("s", "u")
    assert fake.calls == 2 and waits == [12.5]


def test_429_default_wait_is_15_seconds(waits):
    _client(FlakyClient(Boom("too many requests", 429))).complete("s", "u")
    assert waits == [15.0]


def test_429_gives_up_after_6_retries(waits):
    fake = FlakyClient(*[Boom("slow down", 429) for _ in range(10)])
    with pytest.raises(Boom):
        _client(fake).complete("s", "u")
    assert fake.calls == 7 and len(waits) == 6


def test_other_errors_are_not_retried(waits):
    fake = FlakyClient(Boom("bad request", 400))
    with pytest.raises(Boom):
        _client(fake).complete("s", "u")
    assert fake.calls == 1 and waits == []


def test_assistant_survives_429_with_retry(store, waits):
    a = RagAssistant(store, _client(FlakyClient(Boom("retry in 2s", 429))), top_k=3, threshold=0.3)
    ans = a.ask("How many sick leave days per year?")
    assert ans.found and waits == [2.0]


def test_assistant_reports_exhausted_rate_limit_cleanly(store, waits):
    fake = FlakyClient(*[Boom("slow down", 429) for _ in range(10)])
    ans = RagAssistant(store, _client(fake), top_k=3, threshold=0.3).ask("How many sick leave days per year?")
    assert not ans.found and ans.reason == "llm_error" and "Boom" in ans.error
