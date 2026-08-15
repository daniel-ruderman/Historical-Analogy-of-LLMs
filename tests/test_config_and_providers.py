"""Configuration, provider abstraction, retry logic and caching."""

from __future__ import annotations

import pytest

from hal.cache import EmbeddingCache, JsonCache
from hal.config import ROLES, Settings, load_settings
from hal.providers import get_embedding_provider, get_llm, get_search_provider
from hal.providers.base import EmbeddingProvider, LLMProvider, SearchProvider
from hal.providers.caching import CachedEmbeddingProvider
from hal.providers.factory import available_providers, register_llm
from hal.providers.mock import MockEmbeddingProvider, MockLLMProvider
from hal.retry import ProviderError, call_with_retry, is_transient


# --- configuration --------------------------------------------------------
def test_roles_default_to_the_shared_model():
    settings = Settings(llm_model="model-x")
    for role in ROLES:
        assert settings.model_for(role) == "model-x"


def test_per_role_model_overrides_shared_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "shared-model")
    monkeypatch.setenv("CRITIC_MODEL", "critic-model")
    settings = load_settings()
    assert settings.model_for("critic") == "critic-model"
    assert settings.model_for("generator") == "shared-model"
    assert settings.model_for("judge") == "shared-model"


def test_environment_drives_providers_and_loop(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("REFINEMENT_ROUNDS", "3")
    monkeypatch.setenv("MAX_CANDIDATES", "10")
    settings = load_settings()
    assert settings.llm_provider == "mock"
    assert settings.embedding_provider == "mock"
    assert settings.refinement_rounds == 3
    assert settings.max_candidates == 10


def test_invalid_numeric_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("REFINEMENT_ROUNDS", "not-a-number")
    assert load_settings().refinement_rounds == 2


def test_no_api_key_is_hardcoded():
    assert Settings().api_key is None


# --- factory --------------------------------------------------------------
def test_factory_returns_provider_instances(offline_settings):
    assert isinstance(get_llm(role="critic"), LLMProvider)
    assert isinstance(get_embedding_provider(cached=False), EmbeddingProvider)
    assert isinstance(get_search_provider(), SearchProvider)


def test_factory_rejects_unknown_provider(offline_settings):
    with pytest.raises(ValueError):
        get_llm(provider="does-not-exist")


def test_a_new_provider_can_be_registered_without_touching_algorithms(offline_settings):
    class CustomLLM(MockLLMProvider):
        name = "custom"

    register_llm("custom", lambda settings, role, model: CustomLLM(model=model))
    provider = get_llm(role="judge", provider="custom")
    assert provider.name == "custom"
    assert "custom" in available_providers()["llm"]


def test_role_model_reaches_the_provider(monkeypatch, offline_settings):
    from hal.config import set_settings

    set_settings(Settings(llm_provider="mock", llm_model="base",
                          role_models={"critic": "critic-model"}))
    assert get_llm(role="critic").model == "critic-model"
    assert get_llm(role="generator").model == "base"


# --- retry ----------------------------------------------------------------
def test_transient_errors_are_retried_with_backoff():
    delays = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: rate limit exceeded")
        return "ok"

    result = call_with_retry(flaky, max_retries=5, base_delay=1.0,
                             sleep=delays.append)
    assert result == "ok"
    assert calls["n"] == 3
    assert len(delays) == 2
    assert delays[1] > delays[0]  # exponential


def test_permanent_errors_fail_fast():
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise RuntimeError("API key not valid. Please pass a valid API key.")

    with pytest.raises(ProviderError):
        call_with_retry(broken, max_retries=4, sleep=lambda _: None)
    assert calls["n"] == 1


def test_retries_are_bounded():
    with pytest.raises(ProviderError):
        call_with_retry(lambda: (_ for _ in ()).throw(RuntimeError("503 unavailable")),
                        max_retries=2, sleep=lambda _: None)


def test_transient_classification():
    assert is_transient(RuntimeError("429 Too Many Requests"))
    assert is_transient(TimeoutError("deadline exceeded"))
    assert not is_transient(RuntimeError("401 unauthorized"))


# --- caching --------------------------------------------------------------
def test_json_cache_roundtrips(tmp_path):
    cache = JsonCache(tmp_path, "demo")
    cache.set("k", {"a": 1})
    assert JsonCache(tmp_path, "demo").get("k") == {"a": 1}


def test_embedding_cache_is_keyed_by_model(tmp_path):
    a = EmbeddingCache(tmp_path, provider="p", model="model-a")
    b = EmbeddingCache(tmp_path, provider="p", model="model-b")
    a.set("text", [1.0, 0.0])
    assert a.get("text") == [1.0, 0.0]
    assert b.get("text") is None       # vectors from different models never mix
    assert a.metadata["model"] == "model-a"


def test_cached_embedding_provider_avoids_repeat_api_calls(offline_settings):
    inner = MockEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, settings=offline_settings)
    first = cached.embed("the same text")
    second = cached.embed("the same text")
    assert first == second
    assert inner.calls == 1
    assert cached.stats()["cache_hits"] == 1


def test_cached_embedding_batch_only_fetches_missing(offline_settings):
    inner = MockEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, settings=offline_settings)
    cached.embed_batch(["a", "b"])
    calls_after_first = inner.calls
    cached.embed_batch(["a", "b", "c"])
    assert inner.calls == calls_after_first + 1


def test_set_many_writes_the_file_once(tmp_path, monkeypatch):
    """Batch writes keep pool embedding O(n) instead of O(n^2)."""
    cache = JsonCache(tmp_path, "batch")
    writes = {"n": 0}
    real_flush = cache._flush

    def counting_flush(data):
        writes["n"] += 1
        real_flush(data)

    monkeypatch.setattr(cache, "_flush", counting_flush)
    cache.set_many({f"k{i}": i for i in range(50)})
    assert writes["n"] == 1
    assert len(JsonCache(tmp_path, "batch")) == 50


def test_embedding_batch_writes_once_not_once_per_vector(tmp_path, offline_settings,
                                                         monkeypatch):
    inner = MockEmbeddingProvider()
    cached = CachedEmbeddingProvider(inner, settings=offline_settings)
    writes = {"n": 0}
    real_flush = cached.cache._store._flush

    def counting_flush(data):
        writes["n"] += 1
        real_flush(data)

    monkeypatch.setattr(cached.cache._store, "_flush", counting_flush)
    cached.embed_batch([f"text-{i}" for i in range(32)])
    assert writes["n"] == 1


def test_cache_write_failure_is_not_fatal(tmp_path, monkeypatch, capsys):
    """A locked cache file (Windows WinError 32) must not kill a run."""
    import hal.cache as cache_module

    cache = JsonCache(tmp_path, "locked")

    def always_locked(src, dst):
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(cache_module.os, "replace", always_locked)
    monkeypatch.setattr(cache_module.time, "sleep", lambda _s: None)

    cache.set("k", "v")                       # must not raise
    assert cache.get("k") == "v"              # value still usable in memory
    assert "cache warning" in capsys.readouterr().out
    assert list(tmp_path.glob("*.tmp")) == []  # no stray temp files left behind


def test_cache_write_recovers_after_a_transient_lock(tmp_path, monkeypatch):
    import hal.cache as cache_module

    cache = JsonCache(tmp_path, "transient")
    real_replace = cache_module.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(32, "locked")
        return real_replace(src, dst)

    monkeypatch.setattr(cache_module.os, "replace", flaky_replace)
    monkeypatch.setattr(cache_module.time, "sleep", lambda _s: None)

    cache.set("k", "v")
    assert JsonCache(tmp_path, "transient").get("k") == "v"   # persisted on retry
    assert list(tmp_path.glob("*.tmp")) == []


def test_successful_write_leaves_no_temp_file(tmp_path):
    cache = JsonCache(tmp_path, "clean")
    cache.set_many({"a": 1})
    cache.set_many({"b": 2})
    assert list(tmp_path.glob("*.tmp")) == []
    assert len(JsonCache(tmp_path, "clean")) == 2


# --- reasoning models (Gemma 4) -------------------------------------------
def test_empty_answer_from_an_exhausted_token_budget_raises(offline_settings):
    """A reasoning model can burn the whole budget on thinking and return ''.

    Returning that empty string silently would corrupt a run, so it must fail.
    """
    from types import SimpleNamespace

    from hal.providers.gemini import GeminiLLMProvider

    class Models:
        def generate_content(self, **kwargs):
            return SimpleNamespace(
                text="",
                candidates=[SimpleNamespace(finish_reason="MAX_TOKENS",
                                            content=None)],
                usage_metadata=SimpleNamespace(thoughts_token_count=61,
                                               candidates_token_count=None),
            )

    provider = GeminiLLMProvider(
        model="gemma-4-31b-it",
        settings=Settings(api_key="k", max_output_tokens=64, max_retries=0),
        client=SimpleNamespace(models=Models()),
    )
    with pytest.raises(ProviderError) as excinfo:
        provider.generate("prompt")
    message = str(excinfo.value)
    assert "MAX_OUTPUT_TOKENS" in message
    assert "61 tokens went to internal reasoning" in message


def test_normal_empty_answer_is_not_turned_into_an_error(offline_settings):
    """An empty answer with a normal finish reason stays an empty answer."""
    from types import SimpleNamespace

    from hal.providers.gemini import GeminiLLMProvider

    class Models:
        def generate_content(self, **kwargs):
            return SimpleNamespace(
                text="", candidates=[SimpleNamespace(finish_reason="STOP",
                                                     content=None)],
                usage_metadata=None)

    provider = GeminiLLMProvider(
        model="gemma-4-31b-it", settings=Settings(api_key="k"),
        client=SimpleNamespace(models=Models()))
    assert provider.generate("prompt") == ""


def test_default_token_budget_leaves_room_for_thinking():
    """Reasoning models need headroom: thinking tokens count against the budget."""
    from hal.project_defaults import PROJECT_DEFAULTS

    assert int(PROJECT_DEFAULTS["MAX_OUTPUT_TOKENS"]) >= 2048


# --- mock LLM behaviour ---------------------------------------------------
def test_mock_llm_honours_stop_sequences():
    llm = MockLLMProvider(responses=["Spanish flu\nsomething else"])
    assert llm.generate("prompt", stop=["\n"]) == "Spanish flu"
