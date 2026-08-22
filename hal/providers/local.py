"""Local LLM provider — Ollama, LM Studio, llama.cpp server, vLLM.

Nothing in this project changes when you switch to a local model: the
algorithms call :func:`hal.providers.factory.get_llm` and never name a vendor.
Setting ``LLM_PROVIDER=local`` and ``LLM_MODEL=qwen3:8b`` is the whole change.

Two API styles are supported:

``ollama`` (default)
    POSTs to ``/api/chat``. This is the only style that lets us set
    **num_ctx**, the context window. That matters here: our worst-case prompt
    is ~8k tokens (the Final Judge, and the Generate/Search revise step with
    critiques + counterexamples + evidence), while Ollama's own default is
    2k-4k. A too-small window truncates the prompt *silently*, which looks
    exactly like a bad model: "no candidates were produced", unparseable JSON.
    We therefore always send num_ctx explicitly.

``openai``
    POSTs to ``/v1/chat/completions``. Portable to LM Studio, vLLM and
    llama.cpp's server, but that API has no num_ctx field -- the context window
    must then be configured on the server itself.

JSON mode: with the ollama style we pass ``format: "json"``, which constrains
decoding so the model *cannot* emit invalid JSON. That directly addresses the
failure mode seen with Gemma 4, where a third of agentic runs died on
unparseable output.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import Settings, get_settings
from ..retry import ProviderError, call_with_retry
from .base import LLMProvider

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_NUM_CTX = 16384
DEFAULT_KEEP_ALIVE = "30m"


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """POST JSON and return the decoded reply. Uses requests when available."""
    body = json.dumps(payload).encode("utf-8")
    try:  # pragma: no cover - depends on the environment
        import requests

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except ImportError:
        pass
    request = Request(url, data=body,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as handle:  # noqa: S310 - local host
        return json.loads(handle.read().decode("utf-8"))


def _friendly_connection_error(exc: BaseException, base_url: str, model: str) -> str:
    text = str(exc)
    if isinstance(exc, (URLError, ConnectionError, TimeoutError, OSError)) and \
            not isinstance(exc, HTTPError):
        return (f"cannot reach a local model server at {base_url}. Is it running? "
                f"Start it with `ollama serve`, then `ollama pull {model}`.")
    if isinstance(exc, HTTPError) and exc.code == 404:
        return (f"{base_url} has no model named {model!r}. "
                f"Pull it first: `ollama pull {model}`.")
    return text


class LocalLLMProvider(LLMProvider):
    """An LLM served from this machine."""

    name = "local"

    def __init__(self, model: Optional[str] = None, settings: Optional[Settings] = None,
                 role: str = "baseline",
                 post: Optional[Callable[[str, Dict[str, Any], float], Dict[str, Any]]] = None):
        self.settings = settings or get_settings()
        self.model = model or self.settings.model_for(role)
        self.role = role
        self.base_url = (self.settings.local_base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_style = (self.settings.local_api_style or "ollama").lower()
        self.num_ctx = self.settings.local_num_ctx or DEFAULT_NUM_CTX
        self.keep_alive = self.settings.local_keep_alive or DEFAULT_KEEP_ALIVE
        self.timeout = self.settings.local_timeout
        self._post = post or _post_json

    # -- payloads ---------------------------------------------------------
    def _ollama_payload(self, prompt, stop, temperature, max_output_tokens,
                        system, json_output) -> Dict[str, Any]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options: Dict[str, Any] = {
            "temperature": temperature,
            # num_ctx is the whole reason this style is the default: without it
            # Ollama silently truncates our ~8k-token prompts.
            "num_ctx": self.num_ctx,
            "num_predict": max_output_tokens,
        }
        if stop:
            options["stop"] = list(stop)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
            # Keep the model resident between calls; the agentic pipeline makes
            # dozens of calls per example and reloading each time dominates.
            "keep_alive": self.keep_alive,
        }
        if json_output:
            # Constrained decoding: invalid JSON becomes impossible.
            payload["format"] = "json"
        think = (self.settings.local_think or "auto").lower()
        if think in ("true", "false"):
            # Reasoning models (Qwen3, ...) think before answering. Ollama keeps
            # that text in a separate field so it never pollutes our JSON, but it
            # costs 3-4x the wall time. Measured on qwen3:8b: 20-24 s with
            # thinking vs 6 s without, for the same short answer. Off by default;
            # turn it on to test whether it improves analogy quality.
            payload["think"] = think == "true"
        return payload

    def _openai_payload(self, prompt, stop, temperature, max_output_tokens,
                        system, json_output) -> Dict[str, Any]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if stop:
            payload["stop"] = list(stop)
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        return payload

    # -- response parsing -------------------------------------------------
    @staticmethod
    def _extract(data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        message = data.get("message")               # ollama /api/chat
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(data.get("response"), str):   # ollama /api/generate
            return data["response"]
        choices = data.get("choices")               # openai-compatible
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                inner = choice.get("message")
                if isinstance(inner, dict) and isinstance(inner.get("content"), str):
                    return inner["content"]
                if isinstance(choice.get("text"), str):
                    return choice["text"]
        return ""

    # -- LLMProvider ------------------------------------------------------
    def generate(self, prompt: str, *, stop: Optional[Sequence[str]] = None,
                 temperature: Optional[float] = None,
                 max_output_tokens: Optional[int] = None,
                 system: Optional[str] = None, json_output: bool = False) -> str:
        temperature = self.settings.temperature if temperature is None else temperature
        max_output_tokens = max_output_tokens or self.settings.max_output_tokens

        if self.api_style == "openai":
            url = f"{self.base_url}/v1/chat/completions"
            payload = self._openai_payload(prompt, stop, temperature,
                                           max_output_tokens, system, json_output)
        else:
            url = f"{self.base_url}/api/chat"
            payload = self._ollama_payload(prompt, stop, temperature,
                                           max_output_tokens, system, json_output)

        def _call() -> Dict[str, Any]:
            try:
                return self._post(url, payload, self.timeout)
            except Exception as exc:  # noqa: BLE001 - normalise transport errors
                message = _friendly_connection_error(exc, self.base_url, self.model)
                if message is not str(exc):
                    raise ProviderError(message) from exc
                raise

        data = call_with_retry(
            _call,
            max_retries=self.settings.max_retries,
            base_delay=self.settings.retry_base_delay,
            max_delay=self.settings.retry_max_delay,
            request_delay=self.settings.request_delay,
            description=f"local.generate({self.model}, role={self.role})",
        )

        self._check_context_use(data, max_output_tokens)
        text = self._extract(data)
        if not text.strip():
            self._raise_if_truncated(data, max_output_tokens)
        if stop and text and self.api_style == "ollama":
            # Ollama honours stop sequences, but strip defensively so both
            # styles behave identically for the paper's prompts.
            for marker in stop:
                index = text.find(marker)
                if index != -1:
                    text = text[:index]
        return text

    # Largest prompt seen this process, for reporting. Truncation is silent, so
    # we watch how close we get to the window rather than hoping.
    peak_prompt_tokens = 0

    def _check_context_use(self, data: Dict[str, Any], max_output_tokens: int) -> None:
        """Fail loudly when a prompt filled (or overflowed) the context window.

        Ollama does not report truncation -- it drops the START of the prompt and
        answers anyway, so the model silently loses the task description and the
        input event while still seeing the trailing "reply with JSON" protocol.
        The result is well-formed JSON about the wrong thing, which would be
        scored as if it were a real answer. Counting the prompt tokens the server
        actually consumed is the only reliable way to notice.
        """
        prompt_tokens = data.get("prompt_eval_count")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            return
        type(self).peak_prompt_tokens = max(type(self).peak_prompt_tokens,
                                            prompt_tokens)
        if prompt_tokens >= self.num_ctx:
            raise ProviderError(
                f"prompt of {prompt_tokens} tokens met or exceeded the context "
                f"window LOCAL_NUM_CTX={self.num_ctx} for {self.model}. The start "
                f"of the prompt was silently discarded, so the answer cannot be "
                f"trusted. Raise LOCAL_NUM_CTX (costs VRAM) or shorten the prompt "
                f"(fewer candidates / ReAct steps)."
            )
        headroom = self.num_ctx - prompt_tokens
        if headroom < min(max_output_tokens, 512):
            raise ProviderError(
                f"prompt of {prompt_tokens} tokens leaves only {headroom} tokens "
                f"of the {self.num_ctx}-token window for the answer. Raise "
                f"LOCAL_NUM_CTX or lower MAX_OUTPUT_TOKENS."
            )

    def _raise_if_truncated(self, data: Dict[str, Any], max_output_tokens: int) -> None:
        """An empty answer is a failure, never a silent empty string."""
        reason = str(data.get("done_reason") or "")
        if reason == "length":
            raise ProviderError(
                f"{self.model} returned no text: the output budget "
                f"({max_output_tokens}) ran out before any answer. Raise "
                f"MAX_OUTPUT_TOKENS."
            )
        raise ProviderError(
            f"{self.model} returned an empty response from {self.base_url} "
            f"(done_reason={reason or 'unknown'}). If prompts are long, check that "
            f"LOCAL_NUM_CTX ({self.num_ctx}) is large enough -- a short context "
            f"window truncates the prompt silently."
        )
