"""Provider abstractions: LLM, embeddings and search."""

from .base import (
    EmbeddingProvider,
    LLMProvider,
    RecordingLLMProvider,
    SearchProvider,
    SearchResult,
)
from .caching import CachedEmbeddingProvider
from .factory import (
    available_providers,
    get_embedding_provider,
    get_llm,
    get_search_provider,
    register_embedding,
    register_llm,
    register_search,
)

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "SearchProvider",
    "SearchResult",
    "RecordingLLMProvider",
    "CachedEmbeddingProvider",
    "get_llm",
    "get_embedding_provider",
    "get_search_provider",
    "register_llm",
    "register_embedding",
    "register_search",
    "available_providers",
]
