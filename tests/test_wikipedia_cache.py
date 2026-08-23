"""A failed Wikipedia call must never be cached as an absent page.

The popular run of 2026-08-22 was invalidated by exactly this: sustained
throttling produced empty results, the empty results were written to the cache,
and from then on real events such as "Vietnam War" were permanently
"unresolvable". 1048 of 1171 cache entries ended up being stored failures, and
every method that ran late in the night lost most of its examples.

Offline: the MediaWiki call is stubbed, nothing touches the network.
"""

from __future__ import annotations

import pytest

from hal.cache import JsonCache
from hal.config import load_settings
from hal.providers.search import WikipediaSearchProvider


class Boom(RuntimeError):
    """Stands in for a timeout / 429 / connection reset."""


def make_provider(tmp_path, responder):
    settings = load_settings(cache_dir=tmp_path, wiki_request_delay=0.0)
    provider = WikipediaSearchProvider(
        settings=settings,
        cache=JsonCache(tmp_path, "wikipedia_test", enabled=True),
    )
    provider._api = responder          # type: ignore[assignment]
    return provider


# --- failures are not remembered ------------------------------------------
def test_a_failed_search_is_not_cached(tmp_path):
    calls = []

    def failing(params):
        calls.append(params)
        raise Boom("429 too many requests")

    provider = make_provider(tmp_path, failing)
    assert provider.search_titles("Vietnam War") == []
    assert provider.search_titles("Vietnam War") == []
    # Both attempts hit the API: the first failure was NOT cached.
    assert len(calls) == 2
    assert provider.failed_calls == 2


def test_a_failed_page_fetch_is_not_cached(tmp_path):
    calls = []

    def failing(params):
        calls.append(params)
        raise Boom("connection reset")

    provider = make_provider(tmp_path, failing)
    assert provider.get_page("Vietnam War") is None
    assert provider.get_page("Vietnam War") is None
    assert len(calls) == 2


def test_a_recovered_call_returns_the_real_page(tmp_path):
    """The exact scenario that broke the run: fail once, then succeed."""
    state = {"first": True}

    def flaky(params):
        if state["first"]:
            state["first"] = False
            raise Boom("timeout")
        return {"query": {"pages": [
            {"title": "Vietnam War", "extract": "The Vietnam War was a conflict."}
        ]}}

    provider = make_provider(tmp_path, flaky)
    assert provider.get_page("Vietnam War") is None      # transient failure
    page = provider.get_page("Vietnam War")              # retried, not poisoned
    assert page is not None
    assert page.title == "Vietnam War"


# --- genuine absence IS remembered ----------------------------------------
def test_a_missing_page_is_cached(tmp_path):
    calls = []

    def missing(params):
        calls.append(params)
        return {"query": {"pages": [{"title": "Nonesuch", "missing": True}]}}

    provider = make_provider(tmp_path, missing)
    assert provider.get_page("Nonesuch") is None
    assert provider.get_page("Nonesuch") is None
    # Wikipedia answered "no such page"; that is a real result worth caching.
    assert len(calls) == 1


def test_an_empty_search_result_is_cached(tmp_path):
    calls = []

    def empty(params):
        calls.append(params)
        return {"query": {"search": []}}

    provider = make_provider(tmp_path, empty)
    assert provider.search_titles("qqqzzz") == []
    assert provider.search_titles("qqqzzz") == []
    assert len(calls) == 1


def test_a_successful_search_is_cached(tmp_path):
    calls = []

    def ok(params):
        calls.append(params)
        return {"query": {"search": [{"title": "Vietnam War"}]}}

    provider = make_provider(tmp_path, ok)
    assert provider.search_titles("Vietnam War") == ["Vietnam War"]
    assert provider.search_titles("Vietnam War") == ["Vietnam War"]
    assert len(calls) == 1
    assert provider.failed_calls == 0
