"""The local (Ollama / OpenAI-compatible) LLM provider.

No server is contacted: the HTTP call is injected, so these tests run offline.
"""

from __future__ import annotations

import pytest

from hal.config import Settings
from hal.providers import get_llm
from hal.providers.local import LocalLLMProvider
from hal.retry import ProviderError


def make_provider(reply=None, settings=None, capture=None, raises=None, **kwargs):
    """A provider whose HTTP POST is replaced by a stub."""
    reply = reply if reply is not None else {"message": {"content": "Spanish flu"}}

    def fake_post(url, payload, timeout):
        if capture is not None:
            capture.update(url=url, payload=payload, timeout=timeout)
        if raises is not None:
            raise raises
        return reply

    settings = settings or Settings(llm_provider="local", llm_model="qwen3:8b",
                                    max_retries=0)
    return LocalLLMProvider(settings=settings, post=fake_post, **kwargs)


# --- the context-window guard --------------------------------------------
def test_num_ctx_is_always_sent():
    """The single most important setting: a short context truncates silently."""
    capture = {}
    make_provider(capture=capture).generate("hello")
    from hal.project_defaults import PROJECT_DEFAULTS

    options = capture["payload"]["options"]
    assert options["num_ctx"] == int(PROJECT_DEFAULTS["LOCAL_NUM_CTX"])
    assert capture["url"].endswith("/api/chat")  # the style that supports num_ctx


def test_num_ctx_is_configurable(monkeypatch):
    from hal.config import load_settings

    monkeypatch.setenv("LOCAL_NUM_CTX", "8192")
    capture = {}
    make_provider(capture=capture, settings=load_settings()).generate("hello")
    assert capture["payload"]["options"]["num_ctx"] == 8192


# Largest prompt observed across a full-settings agentic example on qwen3:8b
# (140 calls): mean 1265, p90 1473, max 6572 tokens.
MEASURED_MAX_PROMPT_TOKENS = 6572


def test_default_context_clears_the_measured_worst_case_prompt():
    """The window must fit the largest prompt we have actually produced,
    with room left for the answer."""
    from hal.project_defaults import PROJECT_DEFAULTS

    num_ctx = int(PROJECT_DEFAULTS["LOCAL_NUM_CTX"])
    assert num_ctx > MEASURED_MAX_PROMPT_TOKENS
    assert num_ctx - MEASURED_MAX_PROMPT_TOKENS >= 1024   # headroom for output


# --- payload shape --------------------------------------------------------
def test_ollama_payload_carries_the_generation_settings():
    capture = {}
    make_provider(capture=capture).generate(
        "prompt", stop=["\n"], temperature=0.3, max_output_tokens=512,
        system="be terse", json_output=True)
    payload = capture["payload"]
    assert payload["model"] == "qwen3:8b"
    assert payload["stream"] is False
    assert payload["format"] == "json"           # constrained decoding
    assert payload["keep_alive"]                 # model stays resident
    assert payload["options"]["temperature"] == 0.3
    assert payload["options"]["num_predict"] == 512
    assert payload["options"]["stop"] == ["\n"]
    assert payload["messages"][0] == {"role": "system", "content": "be terse"}
    assert payload["messages"][-1]["content"] == "prompt"


def test_json_mode_is_off_unless_requested():
    capture = {}
    make_provider(capture=capture).generate("prompt")
    assert "format" not in capture["payload"]


def test_openai_style_uses_the_other_endpoint_and_fields():
    settings = Settings(llm_provider="local", llm_model="qwen3:8b",
                        local_api_style="openai", max_retries=0)
    capture = {}
    provider = make_provider(
        capture=capture, settings=settings,
        reply={"choices": [{"message": {"content": "Cold War"}}]})
    assert provider.generate("prompt", json_output=True) == "Cold War"
    assert capture["url"].endswith("/v1/chat/completions")
    payload = capture["payload"]
    assert payload["response_format"] == {"type": "json_object"}
    assert "options" not in payload              # not an Ollama payload


# --- response parsing -----------------------------------------------------
@pytest.mark.parametrize("reply,expected", [
    ({"message": {"content": "A"}}, "A"),                        # ollama chat
    ({"response": "B"}, "B"),                                    # ollama generate
    ({"choices": [{"message": {"content": "C"}}]}, "C"),         # openai chat
    ({"choices": [{"text": "D"}]}, "D"),                         # openai completion
])
def test_understands_every_common_response_shape(reply, expected):
    assert make_provider(reply=reply).generate("p") == expected


def test_stop_sequences_are_enforced_locally():
    provider = make_provider(reply={"message": {"content": "Spanish flu\nextra"}})
    assert provider.generate("p", stop=["\n"]) == "Spanish flu"


# --- failure handling -----------------------------------------------------
def test_empty_response_raises_instead_of_returning_nothing():
    with pytest.raises(ProviderError) as excinfo:
        make_provider(reply={"message": {"content": "   "}}).generate("p")
    assert "LOCAL_NUM_CTX" in str(excinfo.value)   # points at the likely cause


def test_output_budget_exhaustion_is_named():
    reply = {"message": {"content": ""}, "done_reason": "length"}
    with pytest.raises(ProviderError) as excinfo:
        make_provider(reply=reply).generate("p")
    assert "MAX_OUTPUT_TOKENS" in str(excinfo.value)


def test_server_not_running_gives_an_actionable_message():
    with pytest.raises(ProviderError) as excinfo:
        make_provider(raises=ConnectionRefusedError("refused")).generate("p")
    message = str(excinfo.value)
    assert "ollama serve" in message
    assert "localhost:11434" in message


# --- context-overflow guard ----------------------------------------------
def test_prompt_that_fills_the_window_raises_instead_of_being_truncated():
    """Ollama drops the START of an oversized prompt and answers anyway.

    That yields well-formed JSON about the wrong event, which would be scored
    as a real answer. The guard must make it an error.
    """
    settings = Settings(llm_provider="local", llm_model="qwen3:8b",
                        local_num_ctx=8192, max_retries=0)
    reply = {"message": {"content": '{"looks":"fine"}'}, "prompt_eval_count": 8192}
    with pytest.raises(ProviderError) as excinfo:
        make_provider(reply=reply, settings=settings).generate("p")
    message = str(excinfo.value)
    assert "silently discarded" in message
    assert "LOCAL_NUM_CTX=8192" in message


def test_prompt_leaving_no_room_for_the_answer_raises():
    settings = Settings(llm_provider="local", llm_model="qwen3:8b",
                        local_num_ctx=8192, max_retries=0)
    reply = {"message": {"content": "hi"}, "prompt_eval_count": 8000}
    with pytest.raises(ProviderError) as excinfo:
        make_provider(reply=reply, settings=settings).generate(
            "p", max_output_tokens=1024)
    assert "leaves only 192 tokens" in str(excinfo.value)


def test_comfortable_prompt_passes_and_is_tracked():
    from hal.providers.local import LocalLLMProvider

    settings = Settings(llm_provider="local", llm_model="qwen3:8b",
                        local_num_ctx=8192, max_retries=0)
    LocalLLMProvider.peak_prompt_tokens = 0
    reply = {"message": {"content": "ok"}, "prompt_eval_count": 3000}
    assert make_provider(reply=reply, settings=settings).generate(
        "p", max_output_tokens=1024) == "ok"
    assert LocalLLMProvider.peak_prompt_tokens == 3000


def test_guard_is_inert_when_the_server_reports_no_count():
    """Other servers may omit prompt_eval_count; that must not break anything."""
    assert make_provider(reply={"message": {"content": "ok"}}).generate("p") == "ok"


# --- thinking mode --------------------------------------------------------
def test_thinking_is_disabled_by_default_for_speed():
    capture = {}
    make_provider(capture=capture).generate("p")
    assert capture["payload"]["think"] is False


def test_thinking_can_be_enabled():
    settings = Settings(llm_provider="local", llm_model="qwen3:8b",
                        local_think="true", max_retries=0)
    capture = {}
    make_provider(capture=capture, settings=settings).generate("p")
    assert capture["payload"]["think"] is True


def test_auto_leaves_the_decision_to_the_model():
    settings = Settings(llm_provider="local", llm_model="qwen3:8b",
                        local_think="auto", max_retries=0)
    capture = {}
    make_provider(capture=capture, settings=settings).generate("p")
    assert "think" not in capture["payload"]


# --- wiring ---------------------------------------------------------------
def test_factory_builds_the_local_provider_for_every_role():
    from hal.config import set_settings

    set_settings(Settings(llm_provider="local", llm_model="qwen3:8b"))
    for role in ("generator", "critic", "anti_analogy", "judge", "summarizer",
                 "baseline", "evaluation"):
        provider = get_llm(role=role)
        assert isinstance(provider, LocalLLMProvider)
        assert provider.model == "qwen3:8b"


def test_ollama_is_an_alias_for_local():
    from hal.config import set_settings

    set_settings(Settings(llm_provider="ollama", llm_model="qwen3:8b"))
    assert isinstance(get_llm(role="judge"), LocalLLMProvider)


def test_switching_to_local_leaves_embeddings_alone(monkeypatch):
    """Going local for generation must not touch the embedding model."""
    from hal.config import load_settings

    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")
    settings = load_settings()
    assert settings.llm_provider == "local"
    assert settings.embedding_provider == "gemini"
    assert settings.embedding_model == "gemini-embedding-001"
