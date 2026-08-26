"""Turn the answer the pipeline *wrote* into the event it *meant*.

The agents describe events; the task asks them to name events. Those are not the
same thing, and the gap is not cosmetic. Measured on the popular run of
2026-08-23, **8 of 20** agentic answers were scored against a Wikipedia page
that was not what the answer named:

    "Abduction of the British hostages in Lebanon"  ->  Lebanon hostage crisis
    "Vietnam War (1955-1975)"                       ->  1955 in the Vietnam War
    "The detention of Uyghur Muslims in China"      ->  Uyghurs

The first of those is the worst case there is: the resolver fell back to a
search, landed on **the input event's own article**, and the answer was scored
against itself for an MDS of 0.00. The second lost a war to a single year of it.

Nothing here changes *which* candidate the pipeline picks on quality grounds --
only which name it hands over, and whether that name turns out to denote the
input event. Both are questions of naming, settled by looking the name up.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from hal.providers.base import SearchProvider

# "Vietnam War (1955-1975)" -> "Vietnam War". Agents append dates and glosses
# habitually, and a parenthetical is enough to miss the article.
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
# A leading article is never part of a MediaWiki title.
_LEADING_ARTICLE = re.compile(r"(?i)^the\s+")


def strip_decoration(name: str) -> str:
    """Remove the decorations agents add to event names."""
    text = (name or "").strip().strip('"').strip()
    previous = None
    while text != previous:                 # "X (a) (b)" needs two passes
        previous = text
        text = _PARENTHETICAL.sub("", text).strip()
    return text.strip().strip(",").strip()


def _title_of(search: SearchProvider, name: str) -> Optional[str]:
    """Canonical title for ``name``, or ``None`` when nothing matches."""
    if not name:
        return None
    resolve = getattr(search, "resolve", None)
    page = resolve(name) if resolve else search.get_page(name)
    if page is None:
        results = search.search(name, top_k=1)
        page = results[0] if results else None
    return page.title if page is not None else None


def canonical_title(search: SearchProvider, name: str) -> Tuple[str, str]:
    """Resolve ``name`` to the title of the page it denotes.

    Returns ``(title, how)`` where ``how`` is ``"exact"`` (the name was already
    a title, possibly after stripping a parenthetical), ``"resolved"`` (it
    denoted a differently-titled page) or ``"unresolved"`` (no page at all, in
    which case ``title`` is the cleaned input).
    """
    cleaned = strip_decoration(name)
    if not cleaned:
        return "", "unresolved"
    for attempt in (cleaned, _LEADING_ARTICLE.sub("", cleaned)):
        title = _title_of(search, attempt)
        if title:
            how = "exact" if title.lower() == attempt.lower() else "resolved"
            return title, how
    return cleaned, "unresolved"


def same_event(search: SearchProvider, name_a: str, name_b: str) -> bool:
    """Whether two names denote the SAME Wikipedia article.

    This is the one restatement test that needs no judgement: if the candidate
    and the input event resolve to one page, they are one event, and comparing
    them is not an analogy. It deliberately does *not* look at the metric's
    literal-similarity term -- that would be tuning the method to its own
    scorer. Two names, one article, therefore not an analogy.
    """
    title_a, how_a = canonical_title(search, name_a)
    title_b, how_b = canonical_title(search, name_b)
    if how_a == "unresolved" or how_b == "unresolved":
        return False                        # cannot claim identity without a page
    return title_a.strip().lower() == title_b.strip().lower()
