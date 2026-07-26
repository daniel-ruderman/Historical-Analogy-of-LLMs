"""Fake providers for tests and offline demos -- no network, no API key.

These make it possible to test the whole pipeline (parsing, refinement loop,
Critic/Anti-Analogy integration, Final Judge, error handling) without spending
a single API call.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from .base import EmbeddingProvider, LLMProvider, SearchProvider, SearchResult

Responder = Callable[[str], str]


class MockLLMProvider(LLMProvider):
    """An LLM whose answers are scripted.

    ``responses`` may be:

    * a list  -- returned in order (the last entry repeats once exhausted);
    * a dict  -- the value of the first key found as a substring of the prompt;
    * a callable -- called with the prompt;
    * ``None`` -- echoes a deterministic placeholder.
    """

    name = "mock"

    def __init__(self,
                 responses: Union[None, Sequence[str], Dict[str, str], Responder] = None,
                 model: str = "mock-model",
                 default: str = ""):
        self.responses = responses
        self.model = model
        self.default = default
        self.prompts: List[str] = []
        self.calls = 0
        self._index = 0

    def generate(self, prompt: str, *, stop=None, temperature=None,
                 max_output_tokens=None, system=None, json_output=False) -> str:
        self.prompts.append(prompt)
        self.calls += 1
        text = self._resolve(prompt)
        if stop:
            for marker in stop:
                index = text.find(marker)
                if index != -1:
                    text = text[:index]
        return text

    def _resolve(self, prompt: str) -> str:
        responses = self.responses
        if responses is None:
            return self.default or f"mock-response-{self.calls}"
        if callable(responses):
            return responses(prompt)
        if isinstance(responses, dict):
            for key, value in responses.items():
                if key in prompt:
                    return value
            return self.default
        sequence = list(responses)
        if not sequence:
            return self.default
        if self._index < len(sequence):
            value = sequence[self._index]
            self._index += 1
            return value
        return sequence[-1]


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embeddings with mild lexical structure.

    Texts sharing content words end up closer together, which is enough to make
    retrieval tests meaningful without any API call.
    """

    name = "mock"

    def __init__(self, model: str = "mock-embedding", dimensions: int = 64):
        self.model = model
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, text: str) -> List[float]:
        self.calls += 1
        vector = [0.0] * self.dimensions
        for token in set(re.findall(r"[a-z0-9]+", (text or "").lower())):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


class MockSearchProvider(SearchProvider):
    """In-memory 'Wikipedia': a title -> description mapping."""

    name = "mock"

    def __init__(self, pages: Optional[Dict[str, str]] = None):
        self.pages = dict(pages or {})
        self.queries: List[str] = []
        self.lookups: List[str] = []

    def add(self, title: str, content: str) -> None:
        self.pages[title] = content

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        self.queries.append(query)
        words = {w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 3}
        scored = []
        for title, content in self.pages.items():
            haystack = f"{title} {content}".lower()
            score = sum(1 for word in words if word in haystack)
            if score:
                scored.append((score, title))
        scored.sort(reverse=True)
        return [self._result(title) for _, title in scored[:top_k]]

    def get_page(self, title: str) -> Optional[SearchResult]:
        self.lookups.append(title)
        if title in self.pages:
            return self._result(title)
        for key in self.pages:
            if key.lower() == (title or "").strip().lower():
                return self._result(key)
        return None

    def _result(self, title: str) -> SearchResult:
        content = self.pages[title]
        return SearchResult(title=title, snippet=content[:200], content=content,
                            url=f"mock://{title}", source="mock")


def make_mock_wikipedia(events: Iterable) -> MockSearchProvider:
    """Build a mock search provider from dataset rows or ``(title, text)`` pairs."""
    pages: Dict[str, str] = {}
    for event in events:
        if isinstance(event, dict):
            title = event.get("event_name") or event.get("history_event_text")
            body = event.get("event_intro") or event.get("history_intro_text") or ""
        else:
            title, body = event
        if title:
            pages[title] = body
    return MockSearchProvider(pages)
