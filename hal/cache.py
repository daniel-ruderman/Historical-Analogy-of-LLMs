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
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


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
        self._warned = False

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
        """Store one entry (rewrites the file -- prefer :meth:`set_many` in loops)."""
        self.set_many({key: value})

    def set_many(self, items: Dict[str, Any]) -> None:
        """Store several entries with a single write.

        The whole namespace is one JSON document, so writing once per batch
        instead of once per item turns an O(n^2) write pattern into O(n) -- it
        matters when the 658-event embedding pool is cached.
        """
        if not self.enabled or not items:
            return
        with self._lock:
            data = self._load()
            data.update(items)
            self._flush(data)

    def _flush(self, data: Dict[str, Any]) -> None:
        """Write the namespace atomically; never raise.

        The cache is an optimisation, so a write failure must not kill a
        research run: the data stays in memory and the next flush retries.
        On Windows ``os.replace`` intermittently fails with ``WinError 32``
        when a virus scanner or indexer momentarily holds the file, hence the
        retries and the per-write unique temporary name.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
        except OSError as exc:
            self._warn(f"could not write cache file: {exc}")
            _unlink_quietly(tmp)
            return

        last_error: Optional[BaseException] = None
        for attempt in range(5):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError as exc:      # Windows: file transiently locked
                last_error = exc
                time.sleep(0.1 * (2 ** attempt))
            except OSError as exc:
                last_error = exc
                break
        _unlink_quietly(tmp)
        self._warn(f"could not update cache file ({last_error}); "
                   "continuing without persisting this batch")

    def _warn(self, message: str) -> None:
        if not self._warned:
            self._warned = True
            print(f"    [cache warning] {self.path.name}: {message}")

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

    def set_many(self, pairs: Sequence[Tuple[str, List[float]]]) -> None:
        """Store several vectors with one file write (used for batch embedding)."""
        self._store.set_many({_hash(text): list(vector) for text, vector in pairs})

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
