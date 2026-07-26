"""Wikipedia access used by the baselines and the agents.

The original repository calls ``wikipedia.summary(entity)`` (with a search
fallback on ``PageError`` and a random pick on ``DisambiguationError``) and
truncates the text at 4096 characters.  :func:`wiki_summary` reproduces that
behaviour on top of a :class:`~hal.providers.base.SearchProvider`, which means

* the same verification procedure (an event that has no Wikipedia entry is
  dropped from the candidate set),
* results are cached, and
* tests can swap in a fake Wikipedia.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .providers.base import SearchProvider


class WikipediaHelper:
    """Thin wrapper giving the baselines the original repository's semantics."""

    def __init__(self, search_provider: SearchProvider, max_chars: int = 4096):
        self.search_provider = search_provider
        self.max_chars = max_chars

    def summary(self, entity: str) -> Optional[str]:
        """Lead-section text for ``entity``; ``None`` when nothing matches.

        Equivalent to ``wikipedia.summary`` plus the original's fallbacks, but
        it returns ``None`` instead of raising -- the original code wraps every
        call in ``try/except: continue``.
        """
        resolve = getattr(self.search_provider, "resolve", None)
        page = resolve(entity) if resolve else self.search_provider.get_page(entity)
        if page is None:
            results = self.search_provider.search(entity, top_k=1)
            page = results[0] if results else None
        if page is None:
            return None
        text = page.content or page.snippet
        if not text:
            return None
        return text[: self.max_chars]

    def exists(self, entity: str) -> bool:
        return self.summary(entity) is not None

    def search_titles(self, entity: str, top_k: int = 10) -> List[str]:
        """Title list, as used by the Pass@1 metric in ``evaluation.py``."""
        search_titles = getattr(self.search_provider, "search_titles", None)
        if search_titles is not None:
            return list(search_titles(entity, top_k))
        return [result.title for result in self.search_provider.search(entity, top_k)]

    def event_dict(self, entity: str) -> Optional[Dict[str, str]]:
        """``{'event_name': ..., 'event_intro': ...}`` or ``None``."""
        intro = self.summary(entity)
        if intro is None:
            return None
        return {"event_name": entity, "event_intro": intro}


def get_wikipedia(search_provider: Optional[SearchProvider] = None,
                  max_chars: Optional[int] = None) -> WikipediaHelper:
    from .config import get_settings
    from .providers.factory import get_search_provider

    settings = get_settings()
    provider = search_provider or get_search_provider(settings=settings)
    return WikipediaHelper(provider, max_chars or settings.wiki_max_chars)
