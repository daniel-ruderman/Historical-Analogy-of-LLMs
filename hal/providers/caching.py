"""Caching decorator for embedding providers.

Embedding the 658-event pool costs 658 API calls; doing that on every run would
be wasteful.  Vectors are cached under a key that includes the provider, model
and dimensionality, so vectors from different embedding models never mix.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..cache import EmbeddingCache
from ..config import Settings, get_settings
from .base import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wrap an :class:`EmbeddingProvider` with an on-disk vector cache."""

    def __init__(self, inner: EmbeddingProvider, settings: Optional[Settings] = None,
                 cache: Optional[EmbeddingCache] = None):
        settings = settings or get_settings()
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.model = inner.model
        self.cache = cache or EmbeddingCache(
            settings.cache_dir,
            provider=inner.name,
            model=inner.model,
            dimensions=getattr(inner, "dimensions", None) or settings.embedding_dimensions,
            enabled=settings.cache_enabled,
        )
        self.api_calls = 0
        self.cache_hits = 0

    def embed(self, text: str) -> List[float]:
        cached = self.cache.get(text)
        if cached is not None:
            self.cache_hits += 1
            return cached
        vector = self.inner.embed(text)
        self.api_calls += 1
        self.cache.set(text, vector)
        return vector

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = []
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        for index, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is None:
                results.append(None)
                missing_indices.append(index)
                missing_texts.append(text)
            else:
                self.cache_hits += 1
                results.append(cached)
        if missing_texts:
            vectors = self.inner.embed_batch(missing_texts)
            self.api_calls += len(missing_texts)
            for index, text, vector in zip(missing_indices, missing_texts, vectors):
                self.cache.set(text, vector)
                results[index] = vector
        return [vector or [] for vector in results]

    def stats(self) -> dict:
        return {
            "model": self.model,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.cache),
            "cache_path": str(self.cache.path),
        }
