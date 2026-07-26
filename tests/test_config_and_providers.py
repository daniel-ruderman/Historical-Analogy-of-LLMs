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


# --- mock LLM behaviour ---------------------------------------------------
def test_mock_llm_honours_stop_sequences():
    llm = MockLLMProvider(responses=["Spanish flu\nsomething else"])
    assert llm.generate("prompt", stop=["\n"]) == "Spanish flu"
