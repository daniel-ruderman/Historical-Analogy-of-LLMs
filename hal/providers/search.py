"""Search providers for the tool-using agents.

:class:`WikipediaSearchProvider` is the default backend: it is free, and the
original paper already uses Wikipedia both to verify that a generated event is
real and to fetch its description.  It talks to the MediaWiki API directly
(``requests``) instead of the ``wikipedia`` PyPI package, because the package
scrapes HTML and is fragile; the *procedure* (search by title, take the first
match, use the lead section as the description) is the same.

Results are cached on disk, so repeated runs do not hit Wikipedia again.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ..cache import JsonCache
from ..config import Settings, get_settings
from ..retry import call_with_retry
from .base import SearchProvider, SearchResult

USER_AGENT = (
    "HistoricalAnalogyResearch/0.1 "
    "(academic research project; https://github.com/Nianqi-Li/Historical-Analogy-of-LLMs)"
)


def _http_get_json(url: str, params: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
    """GET a JSON document. Uses ``requests`` when installed, else urllib."""
    query = f"{url}?{urlencode(params)}"
    try:  # pragma: no cover - depends on the environment
        import requests

        response = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        return response.json()
    except ImportError:
        pass
    request = Request(query, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as handle:  # noqa: S310 - fixed host
        return json.loads(handle.read().decode("utf-8"))


class WikipediaSearchProvider(SearchProvider):
    """Wikipedia/MediaWiki search + page retrieval."""

    name = "wikipedia"

    def __init__(self, settings: Optional[Settings] = None, lang: Optional[str] = None,
                 cache: Optional[JsonCache] = None):
        self.settings = settings or get_settings()
        self.lang = lang or self.settings.wiki_lang
        self.api_url = f"https://{self.lang}.wikipedia.org/w/api.php"
        self._cache = cache if cache is not None else JsonCache(
            self.settings.cache_dir, f"wikipedia_{self.lang}",
            enabled=self.settings.cache_enabled,
        )

    # -- internals --------------------------------------------------------
    def _api(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params.setdefault("format", "json")
        params.setdefault("formatversion", 2)
        return call_with_retry(
            lambda: _http_get_json(self.api_url, params),
            max_retries=min(self.settings.max_retries, 3),
            base_delay=1.0,
            max_delay=15.0,
            description="wikipedia api call",
        )

    # -- SearchProvider ---------------------------------------------------
    def search_titles(self, query: str, top_k: int = 5) -> List[str]:
        """Titles matching ``query`` (same role as ``wikipedia.search``)."""
        key = f"search::{top_k}::{query}"
        cached = self._cache.get(key)
        if cached is not None:
            return list(cached)
        try:
            data = self._api({
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": max(1, top_k),
            })
            titles = [hit["title"] for hit in data.get("query", {}).get("search", [])]
        except Exception:
            titles = []
        self._cache.set(key, titles)
        return titles

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        results: List[SearchResult] = []
        for title in self.search_titles(query, top_k):
            page = self.get_page(title)
            if page is not None:
                results.append(page)
            if len(results) >= top_k:
                break
        return results

    def get_page(self, title: str) -> Optional[SearchResult]:
        """Lead-section extract for ``title`` (``None`` when missing)."""
        if not title or not title.strip():
            return None
        title = title.strip()
        key = f"page::{title}"
        cached = self._cache.get(key)
        if cached is not None:
            if cached == {}:
                return None
            return SearchResult(**cached)
        try:
            data = self._api({
                "action": "query", "prop": "extracts", "exintro": 1,
                "explaintext": 1, "redirects": 1, "titles": title,
            })
            pages = data.get("query", {}).get("pages", [])
            if isinstance(pages, dict):  # formatversion=1 shape
                pages = list(pages.values())
            page = pages[0] if pages else None
        except Exception:
            page = None
        if not page or page.get("missing") or not page.get("extract"):
            self._cache.set(key, {})
            return None
        extract = page["extract"].strip()
        if len(extract) > self.settings.wiki_max_chars:
            extract = extract[: self.settings.wiki_max_chars]
        result = SearchResult(
            title=page.get("title", title),
            snippet=extract[:300],
            content=extract,
            url=f"https://{self.lang}.wikipedia.org/wiki/{quote(page.get('title', title).replace(' ', '_'))}",
            source="wikipedia",
        )
        self._cache.set(key, result.__dict__)
        return result

    def resolve(self, title: str) -> Optional[SearchResult]:
        """Page for ``title``, or for the best search hit when it does not exist.

        Mirrors the fallback in the original ``evaluation.py``: try the exact
        title first, then the first result of a title search.
        """
        page = self.get_page(title)
        if page is not None:
            return page
        for candidate in self.search_titles(title, top_k=3):
            page = self.get_page(candidate)
            if page is not None:
                return page
        return None


class NullSearchProvider(SearchProvider):
    """A provider that finds nothing -- used to run agents without any tool."""

    name = "null"

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        return []

    def get_page(self, title: str) -> Optional[SearchResult]:
        return None
