"""Abstract provider interfaces.

Every algorithm in this project (baselines and our agentic pipeline) depends on
these interfaces only -- never on a concrete SDK.  Swapping Gemini for another
API means writing one new subclass and registering it in
:mod:`hal.providers.factory`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class SearchResult:
    """One document returned by a :class:`SearchProvider`."""

    title: str
    snippet: str = ""
    content: str = ""
    url: str = ""
    source: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "source": self.source,
        }

    def brief(self, max_chars: int = 600) -> str:
        body = self.content or self.snippet
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "..."
        return f"{self.title}: {body}" if body else self.title


class LLMProvider(ABC):
    """Text-in / text-out language model."""

    name: str = "llm"
    model: str = ""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        stop: Optional[Sequence[str]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        system: Optional[str] = None,
        json_output: bool = False,
    ) -> str:
        """Return the model's completion for ``prompt``.

        ``stop``          -- stop sequences (the paper's prompts rely on these)
        ``json_output``   -- ask the provider for a JSON response when it can
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r}>"


class EmbeddingProvider(ABC):
    """Dense text embeddings used by the retrieval baselines."""

    name: str = "embedding"
    model: str = ""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text."""

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed several texts (override when the SDK supports batching)."""
        return [self.embed(text) for text in texts]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} model={self.model!r}>"


class SearchProvider(ABC):
    """External knowledge source for the ReAct-style agents.

    Wikipedia is the default because the original paper already uses it; a web
    search backend can be added later without touching the agents.
    """

    name: str = "search"

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Return up to ``top_k`` results for ``query``."""

    @abstractmethod
    def get_page(self, title: str) -> Optional[SearchResult]:
        """Return the full entry for ``title`` (``None`` when it does not exist)."""

    def exists(self, title: str) -> bool:
        """Whether ``title`` names a real entry -- used for event verification."""
        return self.get_page(title) is not None


@dataclass
class LLMCall:
    """A recorded provider call (for the research logs; no hidden reasoning)."""

    role: str
    prompt_chars: int
    response_chars: int
    model: str
    extra: Dict[str, Any] = field(default_factory=dict)


class RecordingLLMProvider(LLMProvider):
    """Decorator that counts calls made through another provider."""

    def __init__(self, inner: LLMProvider, role: str = "llm"):
        self.inner = inner
        self.role = role
        self.model = inner.model
        self.name = inner.name
        self.calls: List[LLMCall] = []

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.inner.generate(prompt, **kwargs)
        self.calls.append(
            LLMCall(
                role=self.role,
                prompt_chars=len(prompt),
                response_chars=len(response or ""),
                model=self.model,
            )
        )
        return response
