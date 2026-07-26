"""Small on-disk caches so that development does not burn API quota.

Two caches are used:

* :class:`JsonCache`      -- generic key/value store (one JSON file per shard),
                             used for Wikipedia pages and search results.
* :class:`EmbeddingCache` -- vectors keyed by provider/model/dimension *and*
                             a hash of the embedded text, so vectors produced
                             by different embedding models never mix.

The cache is an engineering convenience only -- it never changes a result, it
only avoids repeating an identical external call.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return slug[:80] or "default"


class JsonCache:
    """A namespaced key/value cache backed by one JSON file per namespace."""

    def __init__(self, root: Path, namespace: str, enabled: bool = True):
        self.root = Path(root)
        self.namespace = _slug(namespace)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._data: Optional[Dict[str, Any]] = None

    @property
    def path(self) -> Path:
        return self.root / f"{self.namespace}.json"

    def _load(self) -> Dict[str, Any]:
        if self._data is None:
            if self.enabled and self.path.exists():
                try:
                    self._data = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    self._data = {}
            else:
                self._data = {}
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        if not self.enabled:
            return default
        with self._lock:
            return self._load().get(key, default)

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            data = self._load()
            data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)

    def __contains__(self, key: str) -> bool:
        return self.enabled and key in self._load()

    def __len__(self) -> int:
        return len(self._load())


class EmbeddingCache:
    """Cache for embedding vectors, keyed by the model that produced them."""

    def __init__(self, root: Path, provider: str, model: str,
                 dimensions: Optional[int] = None, enabled: bool = True):
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        namespace = f"emb__{provider}__{model}__{dimensions or 'default'}"
        self._store = JsonCache(Path(root) / "embeddings", namespace, enabled=enabled)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
        }

    def get(self, text: str) -> Optional[List[float]]:
        value = self._store.get(_hash(text))
        if isinstance(value, list):
            return [float(x) for x in value]
        return None

    def set(self, text: str, vector: List[float]) -> None:
        self._store.set(_hash(text), list(vector))

    def __len__(self) -> int:
        return len(self._store)

    @property
    def path(self) -> Path:
        return self._store.path


def make_cache(namespace: str, settings=None) -> JsonCache:
    """Build a :class:`JsonCache` from the global settings."""
    from .config import get_settings

    settings = settings or get_settings()
    return JsonCache(settings.cache_dir, namespace, enabled=settings.cache_enabled)
