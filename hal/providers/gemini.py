"""Gemini implementations of :class:`LLMProvider` / :class:`EmbeddingProvider`.

Uses the current official SDK, ``google-genai``::

    pip install google-genai
    from google import genai
    client = genai.Client(api_key=...)      # or GEMINI_API_KEY in the env

The Gemini API currently exposes two generation surfaces:

* ``client.models.generate_content(...)`` -- the long-standing surface; supports
  ``stop_sequences`` and ``response_mime_type`` directly.
* ``client.interactions.create(...)``     -- the newer Interactions API, which
  Google now recommends for the latest models.

``GEMINI_API_SURFACE`` selects one (``models`` / ``interactions``); the default
``auto`` uses ``models.generate_content`` when the installed SDK exposes it and
falls back to ``interactions.create`` otherwise.  This adapter layer is
deliberately separated from the research code: no method knows which surface
was used.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

from ..config import Settings, get_settings
from ..retry import ProviderError, call_with_retry
from .base import EmbeddingProvider, LLMProvider

# The original repository disables all safety filters because historical events
# (wars, atrocities) otherwise trigger blocks.  We keep that behaviour.
_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)


def _import_genai():
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ProviderError(
            "The Gemini provider requires the official SDK: pip install google-genai"
        ) from exc
    return genai, types


def _make_client(settings: Settings):
    genai, _ = _import_genai()
    api_key = settings.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    if not api_key:
        raise ProviderError(
            "No API key found. Set GEMINI_API_KEY in your environment or .env "
            "(see .env.example). Keys are never stored in this repository."
        )
    return genai.Client(api_key=api_key)


def _raise_if_truncated(response: Any, model: str, max_output_tokens: int) -> None:
    """Turn an empty answer caused by an exhausted token budget into a real error.

    Reasoning ("thinking") models -- the Gemma 4 family among them -- spend
    output tokens on internal reasoning before emitting any text, and
    ``max_output_tokens`` covers *both*. When the budget runs out first the API
    returns a candidate with no text and ``finish_reason=MAX_TOKENS``. Returning
    "" there would silently corrupt a run, so fail loudly with a fix.
    """
    reason = _finish_reason(response)
    if "MAX_TOKENS" not in reason.upper():
        return
    usage = getattr(response, "usage_metadata", None)
    thoughts = getattr(usage, "thoughts_token_count", None) if usage else None
    detail = f" ({thoughts} tokens went to internal reasoning)" if thoughts else ""
    raise ProviderError(
        f"{model} returned no text: the output token budget "
        f"({max_output_tokens}) was exhausted before any answer was produced"
        f"{detail}. Raise MAX_OUTPUT_TOKENS -- reasoning models need headroom "
        f"for thinking tokens on top of the answer."
    )


def _safety_settings(types) -> Optional[list]:
    try:
        return [
            types.SafetySetting(category=category, threshold="BLOCK_NONE")
            for category in _SAFETY_CATEGORIES
        ]
    except Exception:  # pragma: no cover - category names may change
        return None


def _finish_reason(response: Any) -> str:
    """The first candidate's finish reason as a plain string ("" if unknown)."""
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            return str(reason)
    return ""


def _response_text(response: Any) -> str:
    """Pull plain text out of whatever the SDK returned."""
    for attribute in ("text", "output_text"):
        value = getattr(response, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
    # Interactions API: walk the output items.
    for attribute in ("output", "model_output", "candidates"):
        items = getattr(response, attribute, None)
        if not items:
            continue
        chunks: List[str] = []
        for item in items if isinstance(items, (list, tuple)) else [items]:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                chunks.append(text)
                continue
            content = getattr(item, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            for part in parts or []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    chunks.append(part_text)
        if chunks:
            return "".join(chunks)
    return ""


class GeminiLLMProvider(LLMProvider):
    """LLM provider backed by the Gemini API."""

    name = "gemini"

    def __init__(self, model: Optional[str] = None, settings: Optional[Settings] = None,
                 client: Any = None, role: str = "baseline"):
        self.settings = settings or get_settings()
        self.model = model or self.settings.model_for(role)
        self.role = role
        self._client = client
        self._surface: Optional[str] = None

    # -- plumbing ---------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            self._client = _make_client(self.settings)
        return self._client

    def _pick_surface(self) -> str:
        configured = (self.settings.gemini_api_surface or "auto").lower()
        if configured in ("models", "interactions"):
            return configured
        if self._surface is None:
            models = getattr(self.client, "models", None)
            self._surface = (
                "models" if models is not None and hasattr(models, "generate_content")
                else "interactions"
            )
        return self._surface

    # -- LLMProvider ------------------------------------------------------
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
        temperature = self.settings.temperature if temperature is None else temperature
        max_output_tokens = max_output_tokens or self.settings.max_output_tokens
        surface = self._pick_surface()

        def _call() -> str:
            if surface == "models":
                return self._generate_models(
                    prompt, stop, temperature, max_output_tokens, system, json_output
                )
            return self._generate_interactions(
                prompt, stop, temperature, max_output_tokens, system, json_output
            )

        return call_with_retry(
            _call,
            max_retries=self.settings.max_retries,
            base_delay=self.settings.retry_base_delay,
            max_delay=self.settings.retry_max_delay,
            request_delay=self.settings.request_delay,
            description=f"gemini.generate({self.model}, role={self.role})",
        )

    def _generate_models(self, prompt, stop, temperature, max_output_tokens,
                         system, json_output) -> str:
        _, types = _import_genai()
        kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if stop:
            kwargs["stop_sequences"] = list(stop)
        if system:
            kwargs["system_instruction"] = system
        if json_output:
            kwargs["response_mime_type"] = "application/json"
        safety = _safety_settings(types)
        if safety:
            kwargs["safety_settings"] = safety
        try:
            config = types.GenerateContentConfig(**kwargs)
        except TypeError:
            # Drop options this SDK version does not know about.
            config = types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=max_output_tokens
            )
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        text = _response_text(response)
        if not text.strip():
            _raise_if_truncated(response, self.model, max_output_tokens)
        return text

    def _generate_interactions(self, prompt, stop, temperature, max_output_tokens,
                               system, json_output) -> str:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if stop:
            generation_config["stop_sequences"] = list(stop)
        if json_output:
            generation_config["response_mime_type"] = "application/json"
        kwargs = {
            "model": self.model,
            "input": prompt,
            "generation_config": generation_config,
        }
        if system:
            kwargs["system_instruction"] = system
        try:
            response = self.client.interactions.create(**kwargs)
        except TypeError:
            response = self.client.interactions.create(model=self.model, input=prompt)
        text = _response_text(response)
        if stop and text:
            # The Interactions API may ignore stop sequences; enforce locally so
            # that the paper's prompts behave identically on both surfaces.
            for marker in stop:
                index = text.find(marker)
                if index != -1:
                    text = text[:index]
        return text


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the Gemini API.

    Vectors are L2-normalised so that an inner product equals cosine
    similarity -- the original repository relies on that property (OpenAI
    ``text-embedding-3-small`` returns unit-norm vectors, Gemini embeddings are
    not necessarily unit-norm when ``output_dimensionality`` is reduced).
    """

    name = "gemini"

    def __init__(self, model: Optional[str] = None, settings: Optional[Settings] = None,
                 client: Any = None, task_type: str = "SEMANTIC_SIMILARITY"):
        self.settings = settings or get_settings()
        self.model = model or self.settings.embedding_model
        self.dimensions = self.settings.embedding_dimensions
        self.task_type = task_type
        self._client = client
        self._supports_task_type = True

    @property
    def client(self):
        if self._client is None:
            self._client = _make_client(self.settings)
        return self._client

    def _config(self, types):
        kwargs = {}
        if self.dimensions:
            kwargs["output_dimensionality"] = self.dimensions
        if self.task_type and self._supports_task_type:
            kwargs["task_type"] = self.task_type
        if not kwargs:
            return None
        try:
            return types.EmbedContentConfig(**kwargs)
        except TypeError:
            self._supports_task_type = False
            kwargs.pop("task_type", None)
            return types.EmbedContentConfig(**kwargs) if kwargs else None

    def _embed_contents(self, contents: List[str]) -> List[List[float]]:
        _, types = _import_genai()
        blank = [i for i, text in enumerate(contents) if not (text or "").strip()]
        if blank:
            # The API rejects an empty Part with an opaque 400; say what is wrong.
            raise ProviderError(
                f"cannot embed empty text (item(s) {blank} of {len(contents)}). "
                "Provide a non-empty string -- e.g. fall back to the event title."
            )

        def _call():
            config = self._config(types)
            kwargs = {"model": self.model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            return self.client.models.embed_content(**kwargs)

        response = call_with_retry(
            _call,
            max_retries=self.settings.max_retries,
            base_delay=self.settings.retry_base_delay,
            max_delay=self.settings.retry_max_delay,
            request_delay=self.settings.request_delay,
            description=f"gemini.embed({self.model})",
        )
        embeddings = getattr(response, "embeddings", None) or []
        vectors: List[List[float]] = []
        for embedding in embeddings:
            values = getattr(embedding, "values", None)
            if values is None and isinstance(embedding, (list, tuple)):
                values = embedding
            vectors.append([float(v) for v in (values or [])])
        if not vectors:
            raise ProviderError(f"gemini.embed({self.model}) returned no vectors")
        from ..vector import l2_normalize

        return [l2_normalize(v) for v in vectors]

    def embed(self, text: str) -> List[float]:
        return self._embed_contents([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            return self._embed_contents(list(texts))
        except ProviderError:
            # Some deployments cap the batch size; fall back to one at a time.
            return [self.embed(text) for text in texts]
