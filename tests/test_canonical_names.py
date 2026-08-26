"""Naming the event, and refusing to call the input event an analogy for itself.

Every case here is taken from the popular run of 2026-08-23, where 8 of 20
agentic answers were scored against a Wikipedia page that was not the one they
named -- including index 17, whose answer resolved to the input event's own
article and scored MDS 0.00.

Offline: a stub search provider, no network.
"""

from __future__ import annotations

import pytest

from agentic_pipeline.canonical_names import (
    canonical_title,
    same_event,
    strip_decoration,
)
from hal.providers.base import SearchResult


class StubSearch:
    """Titles keyed by exact name, plus a deliberately loose search fallback."""

    def __init__(self, pages, fallback=None):
        self.pages = {k.lower(): k for k in pages}
        self.fallback = fallback or {}

    def _result(self, title):
        return SearchResult(title=title, snippet="", content="an event",
                            url="", source="stub")

    def resolve(self, name):
        hit = self.pages.get((name or "").lower())
        return self._result(hit) if hit else None

    def get_page(self, name):
        return self.resolve(name)

    def search(self, query, top_k=5):
        hit = self.fallback.get((query or "").lower())
        return [self._result(hit)] if hit else []


# --- decoration -----------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Vietnam War (1955-1975)", "Vietnam War"),
    ("Soviet-Afghan War (1979-1989)", "Soviet-Afghan War"),
    ("The Space Race (1957-1975)", "The Space Race"),
    ("Iraq War", "Iraq War"),
    ('  "Cuban Revolution"  ', "Cuban Revolution"),
    ("Rise of the Nazi Party (Germany) (1920s)", "Rise of the Nazi Party"),
])
def test_strip_decoration(raw, expected):
    assert strip_decoration(raw) == expected


# --- resolving to the page the answer meant -------------------------------
def test_a_name_that_is_already_a_title_is_untouched():
    search = StubSearch(["Vietnam War"])
    assert canonical_title(search, "Vietnam War") == ("Vietnam War", "exact")


def test_a_parenthetical_date_no_longer_loses_the_article():
    """The real failure: "Vietnam War (1955-1975)" scored against
    "1955 in the Vietnam War", an article about a single year."""
    search = StubSearch(["Vietnam War"],
                        fallback={"vietnam war (1955-1975)": "1955 in the Vietnam War"})
    title, how = canonical_title(search, "Vietnam War (1955-1975)")
    assert title == "Vietnam War"
    assert how == "exact"


def test_a_leading_article_is_tried_without_it():
    search = StubSearch(["Space Race"])
    assert canonical_title(search, "The Space Race (1957-1975)") == ("Space Race", "exact")


def test_a_description_resolves_to_the_event_it_denotes():
    search = StubSearch(["Iran hostage crisis"],
                        fallback={"abduction of the american hostages in iran":
                                  "Iran hostage crisis"})
    title, how = canonical_title(
        search, "Abduction of the American hostages in Iran")
    assert title == "Iran hostage crisis"
    assert how == "resolved"


def test_a_name_with_no_page_is_reported_unresolved():
    search = StubSearch(["Vietnam War"])
    title, how = canonical_title(search, "The Great Widget Uprising of 1817")
    assert how == "unresolved"
    assert title == "The Great Widget Uprising of 1817"


# --- the restatement test -------------------------------------------------
def test_two_names_for_one_article_are_the_same_event():
    """Index 17: the answer resolved to the input event's own page."""
    search = StubSearch(["Lebanon hostage crisis"],
                        fallback={"abduction of the british hostages in lebanon":
                                  "Lebanon hostage crisis"})
    assert same_event(search, "Abduction of the British hostages in Lebanon",
                      "Lebanon hostage crisis") is True


def test_different_articles_are_different_events():
    search = StubSearch(["Iran hostage crisis", "Lebanon hostage crisis"])
    assert same_event(search, "Iran hostage crisis", "Lebanon hostage crisis") is False


def test_identity_is_never_claimed_without_a_page():
    """An unresolvable name must not be declared the same event by default."""
    search = StubSearch(["Lebanon hostage crisis"])
    assert same_event(search, "Some Invented Event", "Lebanon hostage crisis") is False


# --- the pipeline uses it to choose the winner ----------------------------
def test_pipeline_skips_a_winner_that_is_the_input_event():
    """End-to-end version of index 17.

    The judge's top pick resolves to the input event's own article, so the
    pipeline must fall through to the next-ranked candidate rather than
    reporting the input event as its own analogy.
    """
    from hal.schemas import CandidateAnalogy, HistoricalEvent, JudgeRanking, RankedCandidate
    from agentic_pipeline.pipeline import AgenticAnalogyPipeline

    event = HistoricalEvent(name="Lebanon hostage crisis", description="Kidnappings.")
    search = StubSearch(
        ["Lebanon hostage crisis", "Iran hostage crisis"],
        fallback={"abduction of the british hostages in lebanon":
                  "Lebanon hostage crisis"},
    )
    pipeline = AgenticAnalogyPipeline(search=search)

    class Result:
        ranking = JudgeRanking(ranking=[
            RankedCandidate(rank=1,
                            event_name="Abduction of the British hostages in Lebanon"),
            RankedCandidate(rank=2, event_name="Iran hostage crisis"),
        ])
        errors: list = []

    result = Result()
    result.errors = []
    row = pipeline._first_distinct(event, result)
    assert row is not None
    assert row.event_name == "Iran hostage crisis"
    assert any("input event" in e for e in result.errors)


def test_pipeline_rewrites_a_descriptive_name_to_its_article():
    from hal.schemas import HistoricalEvent, JudgeRanking, RankedCandidate
    from agentic_pipeline.pipeline import AgenticAnalogyPipeline

    event = HistoricalEvent(name="Lebanon hostage crisis", description="Kidnappings.")
    search = StubSearch(
        ["Iran hostage crisis", "Lebanon hostage crisis"],
        fallback={"abduction of the american hostages in iran": "Iran hostage crisis"},
    )
    pipeline = AgenticAnalogyPipeline(search=search)

    class Result:
        ranking = JudgeRanking(ranking=[
            RankedCandidate(rank=1,
                            event_name="Abduction of the American hostages in Iran"),
        ])
        errors: list = []

    result = Result()
    result.errors = []
    row = pipeline._first_distinct(event, result)
    assert row.event_name == "Iran hostage crisis"
