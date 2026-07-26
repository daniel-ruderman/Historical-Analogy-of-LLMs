"""Retry with exponential backoff for transient API failures.

Free-tier Gemini quotas produce 429 responses regularly, so every outbound call
goes through :func:`call_with_retry`.  Errors that are clearly permanent
(bad API key, unknown model, malformed request) are raised immediately -- there
is no point in burning quota on them.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# Substrings that indicate a *temporary* failure worth retrying.
TRANSIENT_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "rate limit",
    "ratelimit",
    "resource_exhausted",
    "resource exhausted",
    "quota",
    "too many requests",
    "deadline",
    "timeout",
    "timed out",
    "temporarily",
    "unavailable",
    "overloaded",
    "internal error",
    "connection reset",
    "connection aborted",
    "remote end closed",
)

# Substrings that indicate a permanent failure -- fail fast.
PERMANENT_MARKERS = (
    "api key not valid",
    "api_key_invalid",
    "invalid api key",
    "permission denied",
    "permission_denied",
    "unauthorized",
    "401",
    "403",
    "not found for api version",
    "is not found",
    "unsupported",
)


class ProviderError(RuntimeError):
    """Raised when a provider call fails permanently or exhausts its retries."""


def is_transient(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in message for marker in PERMANENT_MARKERS):
        return False
    if any(marker in message for marker in TRANSIENT_MARKERS):
        return True
    # Network-level errors without a helpful message are treated as transient.
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Honour a server-provided ``retryDelay``/``Retry-After`` when present."""
    import re

    message = str(exc)
    match = re.search(r"retry[_\-\s]?(?:delay|after)\D{0,12}?(\d+(?:\.\d+)?)", message, re.I)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def call_with_retry(
    func: Callable[[], T],
    *,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    request_delay: float = 0.0,
    description: str = "provider call",
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> T:
    """Call ``func`` retrying transient errors with exponential backoff + jitter."""
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            if request_delay > 0 and attempt == 0:
                sleep(request_delay)
            return func()
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise anything
            last_exc = exc
            if attempt >= max_retries or not is_transient(exc):
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            server_delay = retry_after_seconds(exc)
            if server_delay is not None:
                delay = min(max(delay, server_delay), max_delay)
            delay += random.uniform(0, min(1.0, delay * 0.25))
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            sleep(delay)
    raise ProviderError(f"{description} failed: {last_exc}") from last_exc
