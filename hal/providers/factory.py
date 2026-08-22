"""Provider factory / registry.

Algorithms ask for ``get_llm(role="critic")`` and never name a vendor.  Adding
another vendor means writing one subclass and one ``register_*`` call -- no
method code changes.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from ..config import Settings, get_settings
from .base import EmbeddingProvider, LLMProvider, SearchProvider
from .caching import CachedEmbeddingProvider

LLMFactory = Callable[[Settings, str, Optional[str]], LLMProvider]
EmbeddingFactory = Callable[[Settings, Optional[str]], EmbeddingProvider]
SearchFactory = Callable[[Settings], SearchProvider]

_LLM_REGISTRY: Dict[str, LLMFactory] = {}
_EMBEDDING_REGISTRY: Dict[str, EmbeddingFactory] = {}
_SEARCH_REGISTRY: Dict[str, SearchFactory] = {}


def register_llm(name: str, factory: LLMFactory) -> None:
    _LLM_REGISTRY[name.lower()] = factory


def register_embedding(name: str, factory: EmbeddingFactory) -> None:
    _EMBEDDING_REGISTRY[name.lower()] = factory


def register_search(name: str, factory: SearchFactory) -> None:
    _SEARCH_REGISTRY[name.lower()] = factory


# --- built-in providers --------------------------------------------------
def _gemini_llm(settings: Settings, role: str, model: Optional[str]) -> LLMProvider:
    from .gemini import GeminiLLMProvider

    return GeminiLLMProvider(model=model, settings=settings, role=role)


def _gemini_embedding(settings: Settings, model: Optional[str]) -> EmbeddingProvider:
    from .gemini import GeminiEmbeddingProvider

    return GeminiEmbeddingProvider(model=model, settings=settings)


def _local_llm(settings: Settings, role: str, model: Optional[str]) -> LLMProvider:
    from .local import LocalLLMProvider

    return LocalLLMProvider(model=model, settings=settings, role=role)


def _mock_llm(settings: Settings, role: str, model: Optional[str]) -> LLMProvider:
    from .mock import MockLLMProvider

    return MockLLMProvider(model=model or f"mock-{role}")


def _mock_embedding(settings: Settings, model: Optional[str]) -> EmbeddingProvider:
    from .mock import MockEmbeddingProvider

    return MockEmbeddingProvider(model=model or "mock-embedding")


def _wikipedia_search(settings: Settings) -> SearchProvider:
    from .search import WikipediaSearchProvider

    return WikipediaSearchProvider(settings=settings)


def _null_search(settings: Settings) -> SearchProvider:
    from .search import NullSearchProvider

    return NullSearchProvider()


def _mock_search(settings: Settings) -> SearchProvider:
    from .mock import MockSearchProvider

    return MockSearchProvider()


register_llm("gemini", _gemini_llm)
register_llm("local", _local_llm)
register_llm("ollama", _local_llm)      # alias
register_llm("mock", _mock_llm)
register_embedding("gemini", _gemini_embedding)
register_embedding("mock", _mock_embedding)
register_search("wikipedia", _wikipedia_search)
register_search("mock", _mock_search)
register_search("none", _null_search)


# --- accessors -----------------------------------------------------------
def get_llm(role: str = "baseline", *, model: Optional[str] = None,
            provider: Optional[str] = None,
            settings: Optional[Settings] = None) -> LLMProvider:
    """Return the LLM configured for ``role``.

    ``GENERATOR_MODEL``/``CRITIC_MODEL``/... override ``LLM_MODEL`` per role.
    """
    settings = settings or get_settings()
    provider = (provider or settings.provider_for(role)).lower()
    if provider not in _LLM_REGISTRY:
        raise ValueError(
            f"Unknown LLM_PROVIDER {provider!r}. Known: {sorted(_LLM_REGISTRY)}"
        )
    return _LLM_REGISTRY[provider](settings, role, model or settings.model_for(role))


def get_embedding_provider(*, model: Optional[str] = None,
                           provider: Optional[str] = None,
                           settings: Optional[Settings] = None,
                           cached: bool = True) -> EmbeddingProvider:
    settings = settings or get_settings()
    provider = (provider or settings.embedding_provider).lower()
    if provider not in _EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER {provider!r}. Known: {sorted(_EMBEDDING_REGISTRY)}"
        )
    inner = _EMBEDDING_REGISTRY[provider](settings, model or settings.embedding_model)
    if cached and settings.cache_enabled:
        return CachedEmbeddingProvider(inner, settings=settings)
    return inner


def get_search_provider(*, provider: Optional[str] = None,
                        settings: Optional[Settings] = None) -> SearchProvider:
    settings = settings or get_settings()
    provider = (provider or settings.search_provider).lower()
    if provider not in _SEARCH_REGISTRY:
        raise ValueError(
            f"Unknown SEARCH_PROVIDER {provider!r}. Known: {sorted(_SEARCH_REGISTRY)}"
        )
    return _SEARCH_REGISTRY[provider](settings)


def available_providers() -> Dict[str, list]:
    return {
        "llm": sorted(_LLM_REGISTRY),
        "embedding": sorted(_EMBEDDING_REGISTRY),
        "search": sorted(_SEARCH_REGISTRY),
    }
